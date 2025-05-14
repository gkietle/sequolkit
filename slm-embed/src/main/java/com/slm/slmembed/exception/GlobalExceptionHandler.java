package com.slm.slmembed.exception;

import com.slm.slmembed.response.ResponseWrapper;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHandler {
    @ExceptionHandler(Exception.class)
    public ResponseWrapper<?> handleUnwantedException(Exception e) {
        return new ResponseWrapper<>()
                .setCode(500)
                .setMessage(e.getMessage());
    }

    @ExceptionHandler(AppException.class)
    public ResponseWrapper<?> handleAppException(AppException e) {
        return new ResponseWrapper<>()
                .setCode(e.getCode())
                .setMessage(e.getMessage());
    }

}