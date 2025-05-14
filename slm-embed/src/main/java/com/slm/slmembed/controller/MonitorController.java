package com.slm.slmembed.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class MonitorController {
    @GetMapping("/health-check")
    public String healthCheck() {
        return "OK";
    }
}
