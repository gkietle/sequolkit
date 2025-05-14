package com.slm.slmembed.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.slm.slmembed.enums.ResponseEnum;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

@Getter
@Setter
@Accessors(chain = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ResponseWrapper<T> {
    private int code;
    private String message;
    private T data;

    public static ResponseWrapper<Void> toResponse(int code, String message) {
        return new ResponseWrapper<Void>()
                .setCode(code)
                .setMessage(message);
    }

    public static ResponseWrapper<Void> toResponse(ResponseEnum responseEnum) {
        return toResponse(responseEnum.getCode(), responseEnum.getMessage());
    }

    public static <T> ResponseWrapper<T> success(T data) {
        return new ResponseWrapper<T>()
                .setCode(ResponseEnum.SUCCESS.getCode())
                .setMessage(ResponseEnum.SUCCESS.getMessage())
                .setData(data);
    }

    public static <T> ResponseWrapper<T> success(ResponseEnum responseEnum, T data) {
        return new ResponseWrapper<T>()
                .setCode(responseEnum.getCode())
                .setMessage(responseEnum.getMessage())
                .setData(data);
    }

    public static ResponseWrapper<Void> success() {
        return success(null);
    }
}
