package com.slm.slmembed.controller;

import com.slm.slmembed.response.ResponseWrapper;
import com.slm.slmembed.dto.DatabaseSchemaDto;
import com.slm.slmembed.request.DbConnectionRequest;
import com.slm.slmembed.request.DbConnectionWithQueryRequest;
import com.slm.slmembed.service.SchemaService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;

import static com.slm.slmembed.enums.ResponseEnum.*;

@RestController
@RequestMapping("/api/v1/db")
@RequiredArgsConstructor
public class DatabaseController {

    private final SchemaService schemaService;

    @PostMapping(value = "/get-schema/sqlite", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseWrapper<DatabaseSchemaDto> getSqliteSchema(@RequestPart(value = "file") MultipartFile file) {
        return ResponseWrapper.success(SQLITE_SCHEMA_RETRIEVED_SUCCESSFULLY,
                schemaService.getDatabaseSchemaSQLite(file));
    }

    @PostMapping(value = "/query/sqlite", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseWrapper<List<Map<String, Object>>> querySqliteDatabase(
            @RequestPart(value = "file") MultipartFile file,
            @RequestPart(value = "query") String query) {
        return ResponseWrapper.success(SQLITE_QUERY_EXECUTED_SUCCESSFULLY,
                schemaService.executeQuerySQLite(file, query));
    }

    @PostMapping("/get-schema")
    public ResponseWrapper<DatabaseSchemaDto> getDatabaseSchema(@Valid @RequestBody DbConnectionRequest request) {
        return ResponseWrapper.success(SCHEMA_RETRIEVED_SUCCESSFULLY,
                schemaService.getDatabaseSchema(request));
    }

    @PostMapping("/query")
    public ResponseWrapper<List<Map<String, Object>>> queryDatabase(@Valid @RequestBody DbConnectionWithQueryRequest request) {
        return ResponseWrapper.success(QUERY_EXECUTED_SUCCESSFULLY,
                schemaService.queryDatabase(request));
    }

    @PostMapping("/test-connection")
    public ResponseWrapper<Void> testDatabaseConnection(@Valid @RequestBody DbConnectionRequest request) {
        schemaService.testDatabaseConnection(request);
        return ResponseWrapper.toResponse(CONNECTION_SUCCESSFUL);
    }

    @PostMapping(value = "/test-connection/sqlite", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseWrapper<Void> testSqliteConnection(@RequestPart(value = "file") MultipartFile file) {
        schemaService.testSqliteConnection(file);
        return ResponseWrapper.toResponse(SQLITE_CONNECTION_SUCCESSFUL);
    }
}