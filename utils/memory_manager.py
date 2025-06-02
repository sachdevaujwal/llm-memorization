class MemoryManager:
    def __init__(self, strategy):
        self.strategy = strategy
        self.memory = []

    def update_memory(self, interaction):
        if self.strategy == 'flush':
            self.memory = []
        elif self.strategy == 'always_add':
            self.memory.append(interaction)
        elif self.strategy == 'retrieve':
            self.memory = self._retrieve_relevant(interaction)
        elif self.strategy == 'compress':
            self.memory = [self._compress_memory()]
        else:
            raise ValueError("Unknown strategy")

    def get_memory(self):
        return self.memory

    def _retrieve_relevant(self, interaction):
        # Placeholder logic
        return self.memory[-3:]  # Last 3 interactions as mock retrieval

    def _compress_memory(self):
        # Placeholder compression
        return " ".join(self.memory[-5:])[:300]  # Simple truncation
