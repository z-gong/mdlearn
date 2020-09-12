class Fingerprint():
    def __init__(self):
        self.bit_count = {}
        self.use_pre_idx_list = None
        self._silent = False

    @property
    def idx_list(self):
        return list(self.bit_count.keys())

    @property
    def bit_list(self):
        return list(self.bit_count.values())

    _encoder_dict = {}

    @classmethod
    def register(cls, encoder):
        cls._encoder_dict[encoder.name] = encoder

    @classmethod
    def get_encoder(cls, name):
        return cls._encoder_dict[name]

    @classmethod
    def get_encoder_names(cls):
        return list(cls._encoder_dict.keys())