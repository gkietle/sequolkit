package com.slm.slmembed.dto;

import lombok.Getter;
import lombok.Setter;

import java.util.List;
import java.util.Map;

/**
 * DTO for database schema information
 */
@Getter
@Setter
public class DatabaseSchemaDto {
    private String database;
    private List<TableDto> tables;
}

