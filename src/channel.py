import numpy as np
from enum import Enum

class Numerology(Enum):
    """5G NR numerologies (subcarrier spacing and slot duration)"""
    MU0 = (15, 1.0)      # 15 kHz, 1 ms slot
    MU1 = (30, 0.5)      # 30 kHz, 0.5 ms slot
    MU2 = (60, 0.25)     # 60 kHz, 0.25 ms slot
    MU3 = (120, 0.125)   # 120 kHz, 0.125 ms slot
    
    def __init__(self, scs_khz, slot_duration_ms):
        self.scs_khz = scs_khz
        self.slot_duration_ms = slot_duration_ms

class SimulationConfig:
    """Simulation configuration parameters"""
    # Time settings
    sim_duration_ms: float = 1000.0  # Total simulation time
    slot_duration_ms: float = 0.125    # Default 0.125ms (numerology 3)
    time_step_ms: float = 0.125        # Simulation resolution
    
    # System settings
    bandwidth_mhz: float = 20.0      # System bandwidth
    numerology: Numerology = Numerology.MU3
    carrier_freq_ghz: float = 7    # Center frequency
    tx_power_dbm: float = 46.0       # gNB transmit power
    noise_figure_db: float = 5.0     # UE noise figure
    thermal_noise_dbm_hz: float = -174.0
    
    # Cell layout
    inter_site_distance_m: float = 500.0
    num_sectors: int = 3
    
    # PHY/MAC
    num_prbs: int = 100              # Physical Resource Blocks
    num_symbols_per_slot: int = 14   # Normal CP
    mimo_layers: int = 4             # MIMO configuration
    
    # Channel model
    scenario: str = "UMa"            # Urban Macro (3GPP TR 38.901)
    
    def __post_init__(self):
        self.slot_duration_ms = self.numerology.slot_duration_ms
        self.scs_khz = self.numerology.scs_khz
        self.subcarriers_per_prb = 12
        self.total_subcarriers = self.num_prbs * self.subcarriers_per_prb

class Channel:
    """3GPP TR 38.901 channel models for system-level simulation"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.scenario = config.scenario
        
        # Scenario-specific parameters (3GPP TR 38.901 Table 7.4.1-1)
        self.params = {
            "UMa": {  # Urban Macro
                "path_loss_los": lambda d, f: 28.0 + 22*np.log10(d) + 20*np.log10(f),
                "path_loss_nlos": lambda d, f: 13.54 + 39.08*np.log10(d) + 20*np.log10(f/5) - 0.6*(f-5),
                "shadow_std_los": 4.0,
                "shadow_std_nlos": 6.0,
                "los_prob": lambda d: min(18/d, 1.0) * (1 - np.exp(-d/36)) + np.exp(-d/36),
                "sf_corr_dist": 50.0,  # Shadow fading correlation distance
            },
            "UMi": {  # Urban Micro
                "path_loss_los": lambda d, f: 32.4 + 21*np.log10(d) + 20*np.log10(f),
                "path_loss_nlos": lambda d, f: 22.4 + 35.3*np.log10(d) + 21.3*np.log10(f) - 0.3*(f-5),
                "shadow_std_los": 4.0,
                "shadow_std_nlos": 7.82,
                "los_prob": lambda d: min(18/d, 1.0) * (1 - np.exp(-d/27)) + np.exp(-d/27),
                "sf_corr_dist": 10.0,
            },
            "RMa": {  # Rural Macro
                "path_loss_los": lambda d, f: 31.94 + 22.0*np.log10(d) + 20*np.log10(f),
                "path_loss_nlos": lambda d, f: max(33.83 + 24.38*np.log10(d) + 20*np.log10(f), 
                                                   self.params["RMa"]["path_loss_los"](d,f)),
                "shadow_std_los": 4.0,
                "shadow_std_nlos": 8.0,
                "los_prob": lambda d: np.exp(-(d-10)/1000),
                "sf_corr_dist": 37.0,
            }
        }
        
        # Small-scale fading parameters (simplified)
        self.delay_spread_ns = 100  # RMS delay spread
        self.doppler_freq = 10      # Hz (at 3 km/h, 3.5 GHz)
        
    def calculate_path_loss(self, distance_m: float, freq_ghz: float, 
                           los_state: bool | None = None) -> float:
        """Calculate path loss in dB"""
        params = self.params[self.scenario]
        
        # Determine LOS state if not provided
        if los_state is None:
            los_prob = params["los_prob"](distance_m)
            los_state = np.random.random() < los_prob
            
        # Calculate basic path loss
        if los_state:
            pl = params["path_loss_los"](distance_m, freq_ghz)
            shadow_std = params["shadow_std_los"]
        else:
            pl = params["path_loss_nlos"](distance_m, freq_ghz)
            shadow_std = params["shadow_std_nlos"]
            
        # Add shadow fading (correlated log-normal)
        shadow_fading = np.random.normal(0, shadow_std)
        
        return pl + shadow_fading
    
    def calculate_sinr(self, rx_power_dbm: float, interference_dbm: float, 
                      noise_dbm: float) -> float:
        """Calculate SINR in dB"""
        # Total interference + noise
        total_interference = 10**(interference_dbm/10) + 10**(noise_dbm/10)
        sinr_linear = 10**(rx_power_dbm/10) / total_interference
        return 10 * np.log10(sinr_linear)
    
    def get_spectral_efficiency(self, sinr_db: float) -> float:
        """
        Map SINR to spectral efficiency using Shannon capacity
        with practical implementation margin (approximate 4-bit/s/Hz max for 64QAM)
        """
        # Shannon limit: C = B * log2(1 + SINR)
        sinr_linear = 10**(sinr_db/10)
        se_shannon = np.log2(1 + sinr_linear)
        
        # Practical modulation/coding schemes limit
        # 64QAM ~ 6 bits/symbol with coding ~ 4-5 bits/s/Hz max
        max_se = 5.0  # bits/s/Hz
        return min(se_shannon * 0.75, max_se)  # 25% implementation margin