from enums.response_enum import ResponseEnum

class AppException(Exception):
    def __init__(self, message_or_enum, code: int = None):
        if isinstance(message_or_enum, ResponseEnum):
            self.message = message_or_enum.message
            self.code = message_or_enum.code
        else:
            self.message = message_or_enum
            self.code = code if code is not None else ResponseEnum.INTERNAL_SERVER_ERROR.code

        super().__init__(self.message) # Pass both message and code
