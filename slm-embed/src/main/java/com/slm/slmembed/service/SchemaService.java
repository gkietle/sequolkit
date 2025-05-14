package com.slm.slmembed.service;

import com.slm.slmembed.dto.DatabaseSchemaDto;
import com.slm.slmembed.dto.TableDto;
import com.slm.slmembed.exception.AppException;
import com.slm.slmembed.request.DbConnectionRequest;
import com.slm.slmembed.request.DbConnectionWithQueryRequest;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import jakarta.annotation.PreDestroy;
import javax.sql.DataSource;
import java.io.File;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static com.slm.slmembed.enums.ResponseEnum.*;

@Service
@Slf4j
@RequiredArgsConstructor
public class SchemaService {

    private static final String MYSQL_DRIVER = "com.mysql.cj.jdbc.Driver";
    private static final String POSTGRESQL_DRIVER = "org.postgresql.Driver";
    private static final String SQLITE_DRIVER = "org.sqlite.JDBC";

    // Connection timeout settings
    private static final int CONNECTION_TIMEOUT = 30000; // 30 seconds
    private static final int IDLE_TIMEOUT = 300000; // 5 minutes
    private static final int MAX_POOL_SIZE = 5;
    private static final int MIN_IDLE = 1;

    // Connection pool storage using a thread-safe map
    private final Map<String, DataSource> connectionPool = new ConcurrentHashMap<>();
    private final ScheduledExecutorService scheduledExecutor = Executors.newSingleThreadScheduledExecutor();
    private final FileService fileService;

    // SQL injection prevention patterns
    private static final String[] BLACKLISTED_PATTERNS = {
            "into outfile", "load_file", "--", "/*", "xp_", "shutdown",
            "drop", "delete", "update", "insert", "exec", "execute",
            "waitfor", "delay"
    };

    @PostConstruct
    public void init() {
        scheduledExecutor.scheduleAtFixedRate(this::cleanupIdleConnections, 10, 10, TimeUnit.MINUTES);
    }

    /**
     * Creates a unique key for identifying connections
     */
    private String createConnectionKey(String url, String username, String driverClassName) {
        return driverClassName + ":" + url + ":" + username;
    }

    /**
     * Gets or creates a DataSource based on connection parameters
     */
    private DataSource getOrCreateDataSource(String url, String username, String password, String driverClassName) {
        String connectionKey = createConnectionKey(url, username, driverClassName);

        // Return existing connection if it exists
        return connectionPool.computeIfAbsent(connectionKey, key -> {
            log.info("Creating new connection for {}", key);
            return createDataSource(url, username, password, driverClassName);
        });
    }

    /**
     * Create a new HikariCP data source with optimal settings
     */
    private DataSource createDataSource(String url, String username, String password, String driverClassName) {
        HikariConfig hikariConfig = new HikariConfig();

        String jdbcUrl = buildJdbcUrl(url, driverClassName);
        hikariConfig.setJdbcUrl(jdbcUrl);

        if (!SQLITE_DRIVER.equals(driverClassName)) {
            hikariConfig.setUsername(username);
            hikariConfig.setPassword(password);
        }

        hikariConfig.setDriverClassName(driverClassName);

        // Configure connection pool settings
        hikariConfig.setMaximumPoolSize(MAX_POOL_SIZE);
        hikariConfig.setMinimumIdle(MIN_IDLE);
        hikariConfig.setIdleTimeout(IDLE_TIMEOUT);
        hikariConfig.setConnectionTimeout(CONNECTION_TIMEOUT);
        hikariConfig.setPoolName("HikariPool-" + driverClassName.substring(driverClassName.lastIndexOf('.') + 1));

        hikariConfig.setConnectionTestQuery("SELECT 1");

        return new HikariDataSource(hikariConfig);
    }

    /**
     * Build JDBC URL based on driver type
     */
    private String buildJdbcUrl(String url, String driverClassName) {
        return switch (driverClassName) {
            case MYSQL_DRIVER -> "jdbc:mysql://" + url;
            case POSTGRESQL_DRIVER -> "jdbc:postgresql://" + url;
            case SQLITE_DRIVER -> "jdbc:sqlite:" + url;
            default -> url;
        };
    }

    /**
     * Clean up idle connections
     */
    private void cleanupIdleConnections() {
        log.info("Checking for idle connections to clean up");

        Iterator<Map.Entry<String, DataSource>> iterator = connectionPool.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, DataSource> entry = iterator.next();
            if (entry.getValue() instanceof HikariDataSource hikariDataSource) {
                if (hikariDataSource.getHikariPoolMXBean().getIdleConnections() == hikariDataSource.getHikariPoolMXBean().getTotalConnections()) {
                    log.info("Closing idle connection pool: {}", entry.getKey());
                    closeDataSource(hikariDataSource);
                    iterator.remove();
                }
            }
        }
    }

    /**
     * Clean up all connections when the service is destroyed
     */
    @PreDestroy
    public void cleanupConnections() {
        log.info("Shutting down schema service and closing all database connections");
        scheduledExecutor.shutdownNow();
        for (DataSource dataSource : connectionPool.values()) {
            closeDataSource(dataSource);
        }
        connectionPool.clear();
    }

    /**
     * Entry point to retrieve database schema
     */
    public DatabaseSchemaDto getDatabaseSchema(DbConnectionRequest request) {
        String dbType = request.getDbType().toLowerCase();

        try {
            return switch (dbType) {
                case "mysql" -> getDatabaseSchemaInternal(request, MYSQL_DRIVER);
                case "postgresql" -> getDatabaseSchemaInternal(request, POSTGRESQL_DRIVER);
                default -> throw new AppException(UNSUPPORTED_DATABASE_TYPE);
            };
        } catch (Exception e) {
            log.error("Failed to retrieve schema: ", e);
            throw new AppException(DATABASE_CONNECTION_ERROR);
        }
    }

    /**
     * Common implementation for MySQL and PostgreSQL schema retrieval
     */
    private DatabaseSchemaDto getDatabaseSchemaInternal(DbConnectionRequest request, String driver) {
        try {
            DataSource dataSource = getOrCreateDataSource(
                    request.getUrl(), request.getUsername(), request.getPassword(), driver);

            try (var connection = dataSource.getConnection()) {
                DatabaseMetaData metaData = connection.getMetaData();
                log.info("Connected to {} {}", metaData.getDatabaseProductName(), metaData.getDatabaseProductVersion());

                DatabaseSchemaDto schema = new DatabaseSchemaDto();
                schema.setDatabase(connection.getCatalog());
                List<TableDto> tables = new ArrayList<>();

                try (ResultSet tablesResultSet = metaData.getTables(connection.getCatalog(), null, "%", new String[]{"TABLE"})) {
                    while (tablesResultSet.next()) {
                        String tableName = tablesResultSet.getString("TABLE_NAME");
                        log.debug("Processing table: {}", tableName);

                        TableDto table = extractTableMetadata(metaData, connection.getCatalog(), tableName);
                        tables.add(table);
                    }
                }

                schema.setTables(tables);

                return schema;
            }
        } catch (SQLException e) {
            log.error("Schema retrieval error: ", e);
            throw new AppException(DATABASE_CONNECTION_ERROR);
        }
    }

    /**
     * Extract table metadata including columns and foreign keys
     */
    private TableDto extractTableMetadata(DatabaseMetaData metaData, String catalog, String tableName) throws SQLException {
        TableDto table = new TableDto();
        table.setTableIdentifier(tableName);

        Map<String, List<Map<String, String>>> foreignKeys = extractForeignKeys(metaData, catalog, tableName);
        List<Map<String, Object>> columns = extractColumns(metaData, catalog, tableName, foreignKeys);

        table.setColumns(columns);
        return table;
    }

    /**
     * Extract foreign key information for a table
     */
    private Map<String, List<Map<String, String>>> extractForeignKeys(DatabaseMetaData metaData, String catalog, String tableName) throws SQLException {
        Map<String, List<Map<String, String>>> foreignKeys = new HashMap<>();

        try (ResultSet fkResult = metaData.getImportedKeys(catalog, null, tableName)) {
            while (fkResult.next()) {
                String columnName = fkResult.getString("FKCOLUMN_NAME");
                Map<String, String> relation = new HashMap<>();
                relation.put("toColumn", fkResult.getString("PKCOLUMN_NAME"));
                relation.put("tableIdentifier", fkResult.getString("PKTABLE_NAME"));
                relation.put("type", "OTM");

                foreignKeys.computeIfAbsent(columnName, k -> new ArrayList<>()).add(relation);
            }
        }

        return foreignKeys;
    }

    /**
     * Extract column information for a table
     */
    private List<Map<String, Object>> extractColumns(DatabaseMetaData metaData, String catalog,
                                                     String tableName, Map<String, List<Map<String, String>>> foreignKeys) throws SQLException {
        List<Map<String, Object>> columnsList = new ArrayList<>();

        try (ResultSet columns = metaData.getColumns(catalog, null, tableName, "%")) {
            while (columns.next()) {
                Map<String, Object> columnMap = new HashMap<>();
                String columnName = columns.getString("COLUMN_NAME");

                columnMap.put("columnIdentifier", columnName);
                columnMap.put("columnType", columns.getString("TYPE_NAME") + "(" + columns.getInt("COLUMN_SIZE") + ")");
                columnMap.put("isPrimaryKey", isPrimaryKey(metaData, catalog, tableName, columnName));
                columnMap.put("columnDescription", Optional.ofNullable(columns.getString("REMARKS")).orElse(""));
                columnMap.put("relations", foreignKeys.getOrDefault(columnName, new ArrayList<>()));

                columnsList.add(columnMap);
            }
        }

        return columnsList;
    }

    private boolean isPrimaryKey(DatabaseMetaData metaData, String catalog, String tableName, String columnName) throws SQLException {
        try (ResultSet primaryKeys = metaData.getPrimaryKeys(catalog, null, tableName)) {
            while (primaryKeys.next()) {
                if (columnName.equals(primaryKeys.getString("COLUMN_NAME"))) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Implementation for SQLite schema retrieval
     */
    public DatabaseSchemaDto getDatabaseSchemaSQLite(MultipartFile file) {
        File tempFile = null;

        try {
            tempFile = fileService.saveToTempFile(file);
            String url = tempFile.getAbsolutePath();

            // For SQLite files, use the file path as the key
            DataSource dataSource = getOrCreateDataSource(url, null, null, SQLITE_DRIVER);

            try (var connection = dataSource.getConnection()) {
                DatabaseMetaData metaData = connection.getMetaData();
                log.info("Connected to {} {}", metaData.getDatabaseProductName(), metaData.getDatabaseProductVersion());

                DatabaseSchemaDto schema = new DatabaseSchemaDto();
                schema.setDatabase(tempFile.getName());
                List<TableDto> tables = new ArrayList<>();

                try (var statement = connection.createStatement();
                     var resultSet = statement.executeQuery(
                             "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")) {
                    while (resultSet.next()) {
                        TableDto table = new TableDto();
                        String tableName = resultSet.getString("name");
                        table.setTableIdentifier(tableName);

                        Map<String, List<Map<String, String>>> foreignKeys = extractSQLiteForeignKeys(connection, tableName);
                        List<Map<String, Object>> columns = extractSQLiteColumns(connection, tableName, foreignKeys);

                        table.setColumns(columns);
                        tables.add(table);
                    }
                }

                schema.setTables(tables);

                return schema;
            }
        } catch (SQLException e) {
            log.error("SQLite schema retrieval error: ", e);
            throw new AppException(DATABASE_CONNECTION_ERROR);
        } finally {
            fileService.safeDeleteFile(tempFile);
        }
    }

    /**
     * Extract foreign key information for a SQLite table
     */
    private Map<String, List<Map<String, String>>> extractSQLiteForeignKeys(java.sql.Connection connection, String tableName) throws SQLException {
        Map<String, List<Map<String, String>>> foreignKeys = new HashMap<>();

        try (var fkStatement = connection.createStatement();
             var fkResult = fkStatement.executeQuery("PRAGMA foreign_key_list(" + tableName + ")")) {
            while (fkResult.next()) {
                String fromColumn = fkResult.getString("from");
                Map<String, String> foreignKeyMap = new HashMap<>();
                foreignKeyMap.put("toColumn", fkResult.getString("to"));
                foreignKeyMap.put("tableIdentifier", fkResult.getString("table"));
                foreignKeyMap.put("type", "OTM");

                foreignKeys.computeIfAbsent(fromColumn, k -> new ArrayList<>()).add(foreignKeyMap);
            }
        }

        return foreignKeys;
    }

    /**
     * Extract column information for a SQLite table
     */
    private List<Map<String, Object>> extractSQLiteColumns(java.sql.Connection connection, String tableName,
                                                           Map<String, List<Map<String, String>>> foreignKeys) throws SQLException {
        List<Map<String, Object>> columnsList = new ArrayList<>();

        try (var pragmaStatement = connection.createStatement();
             var pragmaResult = pragmaStatement.executeQuery("PRAGMA table_info(" + tableName + ")")) {
            while (pragmaResult.next()) {
                Map<String, Object> columnMap = new HashMap<>();
                String columnName = pragmaResult.getString("name");
                columnMap.put("columnIdentifier", columnName);
                columnMap.put("columnType", pragmaResult.getString("type"));
                columnMap.put("isPrimaryKey", pragmaResult.getInt("pk") == 1);
                columnMap.put("columnDescription",
                        Optional.ofNullable(pragmaResult.getString("dflt_value")).orElse(""));
                columnMap.put("relations", foreignKeys.getOrDefault(columnName, new ArrayList<>()));

                columnsList.add(columnMap);
            }
        }

        return columnsList;
    }

    /**
     * Entry point for executing queries on databases
     */
    public List<Map<String, Object>> queryDatabase(DbConnectionWithQueryRequest request) {
        String dbType = request.getDbType().toLowerCase();
        String query = request.getQuery();

        // Validate query
        if (isNotValidSQLQuery(query)) {
            throw new AppException(INVALID_QUERY);
        }

        // Execute query based on database type
        return switch (dbType) {
            case "mysql" -> executeQuery(request, query, MYSQL_DRIVER);
            case "postgresql" -> executeQuery(request, query, POSTGRESQL_DRIVER);
            default -> throw new AppException(UNSUPPORTED_DATABASE_TYPE);
        };
    }

    /**
     * Common implementation for executing queries (MySQL/PostgreSQL)
     */
    private List<Map<String, Object>> executeQuery(DbConnectionRequest request, String query, String driver) {
        List<Map<String, Object>> result = new ArrayList<>();

        try {
            DataSource dataSource = getOrCreateDataSource(
                    request.getUrl(), request.getUsername(), request.getPassword(), driver);

            try (var connection = dataSource.getConnection();
                 var statement = connection.createStatement();
                 var resultSet = statement.executeQuery(query)) {

                int columnCount = resultSet.getMetaData().getColumnCount();

                while (resultSet.next()) {
                    Map<String, Object> row = new HashMap<>();
                    for (int i = 1; i <= columnCount; i++) {
                        row.put(resultSet.getMetaData().getColumnName(i), resultSet.getObject(i));
                    }
                    result.add(row);
                }
            }

            return result;
        } catch (SQLException e) {
            log.error("SQL query execution error: ", e);
            throw new AppException(SQL_ERROR, e.getMessage());
        }
    }

    public List<Map<String, Object>> executeQuerySQLite(MultipartFile file, String query) {
        if (isNotValidSQLQuery(query)) {
        }

        List<Map<String, Object>> result = new ArrayList<>();
        File tempFile = null;

        try {
            // Save the uploaded file to a temporary directory.
            tempFile = fileService.saveToTempFile(file);

            // Create the SQLite connection URL using the temporary file's absolute path.
            String url = tempFile.getAbsolutePath();
            DataSource dataSource = getOrCreateDataSource(url, null, null, SQLITE_DRIVER);

            try (var connection = dataSource.getConnection();
                 var statement = connection.createStatement();
                 var resultSet = statement.executeQuery(query)) {

                int columnCount = resultSet.getMetaData().getColumnCount();

                while (resultSet.next()) {
                    Map<String, Object> row = new HashMap<>();
                    for (int i = 1; i <= columnCount; i++) {
                        row.put(resultSet.getMetaData().getColumnName(i), resultSet.getObject(i));
                    }
                    result.add(row);
                }
            }

            return result;
        } catch (SQLException e) {
            log.error("SQLite query execution error: ", e);
            throw new AppException(SQL_ERROR, e.getMessage());
        } finally {
            // Delete the temporary file after processing but keep connection in pool
            fileService.safeDeleteFile(tempFile);
        }
    }

    /**
     * Validate SQL query for security
     * Implements multiple layers of SQL injection prevention
     */
    private boolean isNotValidSQLQuery(String query) {
        if (query == null || query.trim().isEmpty()) {
            return true;
        }

        String normalizedQuery = query.trim().toLowerCase();

        // 1. Check for common SQL injection patterns
        for (String pattern : BLACKLISTED_PATTERNS) {
            if (normalizedQuery.contains(pattern)) {
                return true;
            }
        }

        // 2. Check for balanced quotes and parentheses
        int singleQuotes = 0;
        int doubleQuotes = 0;
        int parentheses = 0;

        for (char c : query.toCharArray()) {
            switch (c) {
                case '\'' -> singleQuotes++;
                case '"' -> doubleQuotes++;
                case '(' -> parentheses++;
                case ')' -> parentheses--;
            }
        }

        return singleQuotes % 2 != 0
                || doubleQuotes % 2 != 0
                || parentheses != 0;
    }

    /**
     * Close the DataSource
     */
    private void closeDataSource(DataSource dataSource) {
        if (dataSource instanceof HikariDataSource) {
            ((HikariDataSource) dataSource).close();
        }
    }

    /**
     * Test database connection without retrieving schema
     */
    public void testDatabaseConnection(DbConnectionRequest request) {
        String dbType = request.getDbType().toLowerCase();
        String driver;

        switch (dbType) {
            case "mysql" -> driver = MYSQL_DRIVER;
            case "postgresql" -> driver = POSTGRESQL_DRIVER;
            default -> throw new AppException(UNSUPPORTED_DATABASE_TYPE);
        }

        try {
            // Create a temporary data source that won't be added to the connection pool
            HikariConfig hikariConfig = getHikariConfig(request, driver);

            try (HikariDataSource dataSource = new HikariDataSource(hikariConfig)) {
                try (var connection = dataSource.getConnection()) {
                    DatabaseMetaData metaData = connection.getMetaData();

                    Map<String, Object> connectionInfo = new HashMap<>();
                    connectionInfo.put("databaseProductName", metaData.getDatabaseProductName());
                    connectionInfo.put("databaseProductVersion", metaData.getDatabaseProductVersion());
                    connectionInfo.put("driverName", metaData.getDriverName());
                    connectionInfo.put("driverVersion", metaData.getDriverVersion());
                    connectionInfo.put("url", metaData.getURL());
                    connectionInfo.put("username", metaData.getUserName());
                    connectionInfo.put("catalog", connection.getCatalog());
                }
            }
        } catch (SQLException e) {
            log.error("Connection test failed: ", e);
            throw new AppException(DATABASE_CONNECTION_ERROR);
        }
    }

    private HikariConfig getHikariConfig(DbConnectionRequest request, String driver) {
        HikariConfig hikariConfig = new HikariConfig();
        String jdbcUrl = buildJdbcUrl(request.getUrl(), driver);
        hikariConfig.setJdbcUrl(jdbcUrl);
        hikariConfig.setUsername(request.getUsername());
        hikariConfig.setPassword(request.getPassword());
        hikariConfig.setDriverClassName(driver);
        hikariConfig.setConnectionTimeout(CONNECTION_TIMEOUT);

        // Reduce connection pool size for test connections
        hikariConfig.setMaximumPoolSize(1);
        hikariConfig.setMinimumIdle(0);
        return hikariConfig;
    }

    /**
     * Test SQLite database connection without retrieving schema
     */
    public void testSqliteConnection(MultipartFile file) {
        File tempFile = null;

        try {
            tempFile = fileService.saveToTempFile(file);
            HikariConfig hikariConfig = getHikariConfig(tempFile);

            try (HikariDataSource dataSource = new HikariDataSource(hikariConfig)) {
                try (var connection = dataSource.getConnection()) {
                    DatabaseMetaData metaData = connection.getMetaData();

                    // Get table count to verify database can be read
                    int tableCount = 0;
                    try (var statement = connection.createStatement();
                         var resultSet = statement.executeQuery(
                                 "SELECT count(name) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")) {
                        if (resultSet.next()) {
                            tableCount = resultSet.getInt(1);
                        }
                    }

                    Map<String, Object> connectionInfo = new HashMap<>();
                    connectionInfo.put("databaseProductName", metaData.getDatabaseProductName());
                    connectionInfo.put("databaseProductVersion", metaData.getDatabaseProductVersion());
                    connectionInfo.put("driverName", metaData.getDriverName());
                    connectionInfo.put("driverVersion", metaData.getDriverVersion());
                    connectionInfo.put("fileName", file.getOriginalFilename());
                    connectionInfo.put("fileSize", file.getSize());
                    connectionInfo.put("tableCount", tableCount);
                }
            }
        } catch (SQLException e) {
            log.error("SQLite connection test failed: ", e);
            throw new AppException(DATABASE_CONNECTION_ERROR);
        } finally {
            // Delete the temporary file after testing
            fileService.safeDeleteFile(tempFile);
        }
    }

    private static HikariConfig getHikariConfig(File tempFile) {
        String url = tempFile.getAbsolutePath();

        // Create a temporary data source that won't be added to the connection pool
        HikariConfig hikariConfig = new HikariConfig();
        hikariConfig.setJdbcUrl("jdbc:sqlite:" + url);
        hikariConfig.setDriverClassName(SQLITE_DRIVER);
        hikariConfig.setConnectionTimeout(CONNECTION_TIMEOUT);

        // Set minimal pool settings for test
        hikariConfig.setMaximumPoolSize(1);
        hikariConfig.setMinimumIdle(0);
        return hikariConfig;
    }

}