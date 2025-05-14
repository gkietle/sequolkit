package com.slm.slmembed.request;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class DbConnectionWithQueryRequest extends DbConnectionRequest{
    @NotBlank(message = "Query is required")
    private String query;
}
