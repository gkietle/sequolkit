package com.slm.slmembed.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.Getter;
import lombok.Setter;
import org.springframework.web.multipart.MultipartFile;


@Getter
@Setter
public class DbConnectionRequest {
    @NotBlank(message = "Database name is required")
    @Pattern(regexp = "^[\\w.-]+:\\d+/[\\w-]+$", message = "URL must be in format hostname:port/database")
    private String url;

    @NotBlank(message = "Username is required")
    private String username;

    @NotBlank(message = "Password is required")
    private String password;

    @NotBlank(message = "Database type is required")
    @Pattern(regexp = "mysql|postgresql", message = "Database type must be mysql, postgresql, or sqlite")
    private String dbType;

}