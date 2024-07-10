import timeit


class TimeHelper:
    """
    A helper class for measuring execution time.

    Attributes:
        start_time (float): The start time of the timer.
        end_time (float): The end time of the timer.
        print_time (bool): Flag indicating whether to print the execution time.
        delta_time (float): The difference between the start and end time.

    Methods:
        start_timer(): Starts the timer and returns the start time.
        end_timer(print_time=True): Ends the timer, calculates the execution time,
            and optionally prints it.

    """

    def __init__(self, print_time: bool = True) -> None:
        self.start_time: float = timeit.default_timer()
        self.end_time: float = None
        self.print_time: float = print_time
        self.delta_time: float = None

    def start_timer(self) -> float:
        """
        Starts the timer and returns the start time.

        Returns:
            float: The start time of the timer.

        """
        self.start_time: float = timeit.default_timer()
        return self.start_time

    def end_timer(self, print_time=True) -> float:
        """
        Ends the timer, calculates the execution time, and optionally prints it.

        Args:
            print_time (bool, optional): Flag indicating whether to print the execution time.
                Defaults to True.

        Returns:
            float: The execution time in seconds.

        """
        self.end_time: float = timeit.default_timer()
        self.delta_time: float = self.end_time - self.start_time
        if print_time:
            print(f"Execution time: {self.delta_time} seconds")
        return self.delta_time
