

class FaildToConvergeWarning(Warning):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)