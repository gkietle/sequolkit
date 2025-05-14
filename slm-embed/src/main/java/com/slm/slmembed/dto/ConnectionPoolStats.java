package com.slm.slmembed.dto;

import lombok.Getter;
import lombok.Setter;

import java.util.Map; /**
 * DTO for connection pool statistics
 */
@Getter
@Setter
public class ConnectionPoolStats {
    private int activeConnections;
    private Map<String, Map<String, Object>> details;
}
