from __future__ import annotations

from .qo_splitter import FeatureQuantizer, QOSplitter, Slot


class AdaptiveQOSplitter(QOSplitter):
    def __init__(
            self,
            kernel: str,
            radius: float = 0.5,
            allow_multiway_splits=True,
            gamma: float = 1.0,
    ):
        super().__init__()
        if radius <= 0:
            raise ValueError("'radius' must be greater than zero.")
        self.radius = radius
        self.kernel = kernel
        self.gamma = gamma

        if kernel == "triangular":
            self._quantizer = TriangularFeatureQuantizer(radius)
        elif kernel == "epanechnikov":
            self._quantizer = EpanechnikovFeatureQuantizer(radius)
        elif kernel == "smooth":
            self._quantizer = SmoothStepFeatureQuantizer(radius, gamma=gamma)

        self.allow_multiway_splits = allow_multiway_splits


class TriangularFeatureQuantizer(FeatureQuantizer):
    """
    Implementação do componente de Quantização Suave do Soft-QO.

    Diferente do FeatureQuantizer padrão, este distribui o peso de uma instância
    entre slots adjacentes baseando-se na distância do valor x ao centro do slot.
    Isso implementa a ideia de 'Mapeamento Probabilístico' descrita em [5].
    """

    def __init__(self, radius: float):
        super().__init__(radius=radius)

    def _get_soft_assignments(self, x: float):
        """
        Calcula os slots e os pesos fracionários para o valor x.

        Lógica baseada na proposta de substituir atribuição binária por distribuição [5].
        Usamos interpolação linear (kernel triangular) para distribuir o peso entre
        o slot atual e o vizinho mais próximo.
        """
        # Posição contínua no grid de quantização
        pos = x / self.radius

        # Slot central (onde cairia no QO tradicional se usássemos round)
        center_idx = round(pos)

        # Distância do centro do slot (-0.5 a +0.5)
        dist = pos - center_idx

        # No kernel triangular, a influência decai linearmente com a distância.
        # O slot central recebe a maior parte, o vizinho recebe o resto.
        # Se dist > 0, o dado está à direita do centro -> vizinho é center_idx + 1
        # Se dist < 0, o dado está à esquerda do centro -> vizinho é center_idx - 1

        weight_center = 1.0 - abs(dist)
        weight_neighbor = abs(dist)

        neighbor_idx = center_idx + 1 if dist > 0 else center_idx - 1

        # Retorna pares (índice, peso)
        # Filtramos pesos muito pequenos para economizar memória (esparsidade)
        assignments = []
        if weight_center > 1e-1:
            assignments.append((center_idx, weight_center))
        if weight_neighbor > 1e-1:
            assignments.append((neighbor_idx, weight_neighbor))

        return assignments

    def update(self, x: float, y, weight: float):
        """
        Atualiza as estatísticas distribuindo o peso da instância entre slots vizinhos.
        Isso simula a adaptação gradual descrita como 'O Pulo do Gato' em [7].
        """
        assignments = self._get_soft_assignments(x)

        for index, soft_weight in assignments:
            # O peso final é o peso da instância multiplicado pelo peso do kernel (soft)
            effective_weight = weight * soft_weight

            try:
                # Atualiza o slot existente
                self.hash[index].update(x, y, effective_weight)
            except KeyError:
                # Cria novo slot se não existir (crescimento dinâmico)
                self.hash[index] = Slot(x, y, effective_weight)


class EpanechnikovFeatureQuantizer(FeatureQuantizer):
    """
    Implementação do Soft-QO usando Kernel de Epanechnikov.
    Vantagem: Suporte compacto (não vaza infinitamente como a Gaussiana)
    e computacionalmente mais leve (apenas multiplicações, sem exp).
    """

    def __init__(self, radius: float, bandwidth_scale: float = 1.0):
        super().__init__(radius)
        self.bandwidth = radius * bandwidth_scale

    def _epanechnikov_kernel(self, u):
        """
        Fórmula padrão do Epanechnikov: K(u) = 0.75 * (1 - u^2) para |u| <= 1
        """
        if abs(u) > 1.0:
            return 0.0
        return 0.75 * (1.0 - u**2)

    def _get_assignments(self, x: float):
        # Centro do slot principal
        pos = x / self.radius
        center_idx = round(pos)

        assignments = []
        weights = []
        total_weight = 0.0

        neighbors = range(center_idx - 1, center_idx + 2)

        for idx in neighbors:
            centroid = idx * self.radius
            distance = x - centroid

            u = distance / self.bandwidth

            w = self._epanechnikov_kernel(u)

            if w > 1e-1:
                weights.append(w)
                assignments.append(idx)
                total_weight += w

        final_assignments = []
        if total_weight > 0:
            max_w = max(weights)
            if max_w / total_weight > 0.9:
                max_idx = assignments[weights.index(max_w)]
                return [(max_idx, 1.0)]

            for i, w in enumerate(weights):
                normalized = w / total_weight
                final_assignments.append((assignments[i], normalized))

        return final_assignments

    def update(self, x: float, y, weight: float):
        assignments = self._get_assignments(x)
        for index, soft_weight in assignments:
            effective_weight = weight * soft_weight
            try:
                self.hash[index].update(x, y, effective_weight)
            except KeyError:
                self.hash[index] = Slot(x, y, effective_weight)


class SmoothStepFeatureQuantizer(FeatureQuantizer):
    """
    Implementação baseada na função de ativação da Soft Hoeffding Tree [SoHoT].
    Fonte: Equation (1) de 'Soft Hoeffding Tree' paper.
    """

    def __init__(self, radius: float, gamma: float = 1.0):
        super().__init__(radius)
        # Gamma controla a suavidade da transição (zona de incerteza).
        # Se gamma for pequeno, comporta-se quase como Hard QO.
        self.gamma = gamma

    def _smooth_step(self, t):
        """
        Implementação exata da Eq. 1 da fonte SoHoT.
        S(t) retorna valores entre 0 e 1.
        """
        gamma_half = self.gamma / 2.0

        if t <= -gamma_half:
            return 0.0
        elif t >= gamma_half:
            return 1.0
        else:
            # Polinômio cúbico para interpolação suave na zona de transição
            # S(t) = -2/(gamma^3) * t^3 + 3/(2*gamma) * t + 1/2
            term1 = -2 * (t**3) / (self.gamma**3)
            term2 = 3 * t / (2 * self.gamma)
            return term1 + term2 + 0.5

    def _get_assignments(self, x: float):
        # Mapeamento para o índice contínuo
        pos = x / self.radius

        # Identifica em qual 'fronteira' estamos.
        # Hard QO corta em indices inteiros +/- 0.5 (ex: 0.5, 1.5, 2.5)
        # Vamos definir a 'decisão' baseada na distância ao centro mais próximo.

        center_idx = round(pos)

        # Distância relativa ao centro (-0.5 a +0.5 unidades de raio)
        # Multiplicamos pelo radius para ter a distância física 't' usada no SmoothStep
        dist_from_center = (pos - center_idx) * self.radius

        # O SmoothStep decide "quanto" o dado pertence ao lado Direito vs Esquerdo?
        # Aqui, adaptamos para: quanto o dado pertence ao centro vs vizinho.

        # Se dist_from_center é 0, t=0 -> S(0) = 0.5 (transição perfeita? Não ideal para QO).
        # ADAPTAÇÃO PARA QO:
        # Queremos que no centro do slot o peso seja 1.0. Nas bordas decaia.
        # Vamos definir t como a distância para a Borda do slot.

        assignments = []

        # Lógica simplificada de "Gating" entre dois slots (k e k+1 ou k-1)
        # Se o dado está à direita do centro (dist > 0)
        if dist_from_center > 0:
            neighbor = center_idx + 1
            # t é a distância da fronteira média (radius/2)
            # Se x está muito perto da fronteira, ativa mistura.
            distance_to_boundary = dist_from_center - (self.radius / 2.0)

        else:  # Dado à esquerda do centro
            neighbor = center_idx - 1
            # Distância para a fronteira esquerda (-radius/2)
            # Invertemos o sinal para manter a lógica do smooth step crescente
            distance_to_boundary = (-dist_from_center) - (self.radius / 2.0)

        transfer_weight = self._smooth_step(distance_to_boundary)

        w_center = 1.0 - transfer_weight
        w_neighbor = transfer_weight

        if w_center > 0:
            assignments.append((center_idx, w_center))
        if w_neighbor > 0:
            assignments.append((neighbor, w_neighbor))

        return assignments

    def update(self, x: float, y, weight: float):
        assignments = self._get_assignments(x)
        for index, soft_weight in assignments:
            if soft_weight <= 1e-1:
                continue  # Otimização de esparsidade

            effective_weight = weight * soft_weight
            try:
                self.hash[index].update(x, y, effective_weight)
            except KeyError:
                self.hash[index] = Slot(x, y, effective_weight)
