package com.slm.slmembed.service;

import com.slm.slmembed.enums.ResponseEnum;
import com.slm.slmembed.exception.AppException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

import static com.slm.slmembed.enums.ResponseEnum.*;

/**
 * Service for handling file operations safely
 */
@Service
@Slf4j
public class FileService {

    public File saveToTempFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            log.warn("Attempted to save an empty or null file");
            throw new AppException(FILE_IS_EMPTY);
        }

        try {
            // Create a temporary file with a unique name
            String originalFilename = file.getOriginalFilename();
            String extension = originalFilename != null ?
                    originalFilename.substring(originalFilename.lastIndexOf(".")) : ".tmp";

            // Generate a random filename to prevent path traversal attacks
            String uniqueFilename = UUID.randomUUID() + extension;

            Path tempFile = Files.createTempFile("upload_", uniqueFilename);
            file.transferTo(tempFile.toFile());
            log.debug("Saved file {} to temp location: {}", originalFilename, tempFile);

            // Register a shutdown hook to ensure the file is deleted on JVM exit
            File physicalFile = tempFile.toFile();
            physicalFile.deleteOnExit();

            return physicalFile;
        } catch (IOException e) {
            throw new AppException(FILE_UPLOAD_ERROR);
        }
    }

    /**
     * Safely delete a file with logging
     *
     * @param file The file to delete
     */
    public void safeDeleteFile(File file) {
        if (file != null && file.exists()) {
            boolean deleted = file.delete();
            if (!deleted) {
                log.warn("Failed to delete temporary file: {}", file.getAbsolutePath());
                // Ensure it's deleted on JVM exit as a fallback
                file.deleteOnExit();
            } else {
                log.debug("Successfully deleted temp file: {}", file.getAbsolutePath());
            }
        }
    }
}