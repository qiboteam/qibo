from enum import Enum

class Modality(Enum):
    GATE = "gate"
    PHOTONIC = "photonic"
    PHOTONIC_CV = "photonic-cv"
    PHOTONIC_DV = "photonic-dv"
    PULSE = "pulse"

    @property
    def is_gate(self) -> bool:
        """Check if the modality is gate-based."""
        return self == Modality.GATE

    @property
    def is_photonic(self) -> bool:
        """Check if the modality is photonic."""
        return self in (Modality.PHOTONIC, Modality.PHOTONIC_DV, Modality.PHOTONIC_CV)

    @property
    def is_pulse(self) -> bool:
        """Check if the modality is pulse."""
        return self == Modality.PULSE

