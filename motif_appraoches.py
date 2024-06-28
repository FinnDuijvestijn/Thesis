def define_motifs(self, cluster_size = 3):
    # cluster motif
    motifs = []
    for layer in range(1, self.n_layers - 1):
        motif = set()
        input_dim = self.dimensions[layer - 1]
        output_dim = self.dimensions[layer]

        for i in range(0, input_dim, cluster_size):
            output_index = i // cluster_size
            if output_index < output_dim:
                for j in range(cluster_size):
                    if i + j < input_dim:
                        motif.add((i + j, output_index))

        motifs.append(motif)
    return motifs


def define_motifs(self, receptive_field_size=3):
    # local receptive field motifs
    motifs = []
    for layer in range(1, self.n_layers - 1):
        motif = set()
        input_dim = self.dimensions[layer - 1]
        output_dim = self.dimensions[layer]
        
        for i in range(input_dim):
            for j in range(max(0, i - receptive_field_size // 2), min(output_dim, i + receptive_field_size // 2 + 1)):
                motif.add((i, j))

        motifs.append(motif)
    return motifs


def define_motifs(self, cluster_size=3, overlap=1):
    # clustered overlap motifs
    motifs = []
    for layer in range(1, self.n_layers - 1):
        motif = set()
        input_dim = self.dimensions[layer - 1]
        output_dim = self.dimensions[layer]

        for i in range(0, input_dim, cluster_size - overlap):
            output_index = i // (cluster_size - overlap)
            if output_index < output_dim:
                for j in range(min(cluster_size, input_dim - i)):
                    motif.add((i + j, output_index))

        motifs.append(motif)
    return motifs


def define_motifs(self):
    # Bidirectional motifs
    motifs = []
    for layer in range(1, self.n_layers - 1):
        motif = set()
        input_dim = self.dimensions[layer - 1]
        output_dim = self.dimensions[layer]
        
        for i in range(input_dim):
            if i < output_dim:
                motif.add((i, i))  # Forward connection
            if i - 1 >= 0 and i - 1 < output_dim:
                motif.add((i, i - 1))  # Backward connection
            if i + 1 < output_dim:
                motif.add((i, i + 1))  # Next connection

        motifs.append(motif)
    return motifs