package com.slm.slmembed.dto;

import lombok.Getter;
import lombok.Setter;

import java.util.List;
import java.util.Map; /**
 * DTO for table metadata
 */
@Getter
@Setter
public class TableDto {
    private String tableIdentifier;
    private List<Map<String, Object>> columns;
}
