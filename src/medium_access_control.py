class TimeDivisionDuplexingTable:
    def __init__(self, subcarrier_spacing: float, guard_period: int) -> None:
        """
        Create TDD table.
        
        :param radio_frame_length: Description
        :type radio_frame_length: int
        :param subcarrier_spacing: Description
        :type subcarrier_spacing: float
        :param guard_period: Description
        :type guard_period: int
        """
        self.radio_frame_length: int = 10 #ms
        self.subframe_length: float = self.radio_frame_length / 10 # 1ms
        self.guard_period: float = 7 * self.subframe_length #
        self.subcarrier_spacing: float = subcarrier_spacing

        self.table: list[tuple[float,int]] = []

    def allocate_resources(self):
        pass
