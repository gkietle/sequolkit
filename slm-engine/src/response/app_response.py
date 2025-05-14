from typing import Generic, TypeVar, Optional
from flask import jsonify
from enums.response_enum import ResponseEnum

T = TypeVar("T")

class ResponseWrapper(Generic[T]):
    def __init__(self, code: int, message: str, data: Optional[T] = None):
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self):
        return {"code": self.code, "message": self.message, "data": self.data}

    def to_json(self):
        return jsonify(self.to_dict())

    @classmethod
    def to_response(cls, code: int, message: str):
        return cls(code, message).to_json()

    @classmethod
    def to_response_enum(cls, response_enum: ResponseEnum):
        return cls(response_enum.code, response_enum.message).to_json()

    @classmethod
    def success(cls, data: Optional[T] = None):
        return cls(ResponseEnum.SUCCESS.code, ResponseEnum.SUCCESS.message, data).to_json()
