package com.capstone.backend.domain.emotion.controller;

import com.capstone.backend.common.response.ApiResponse;
import com.capstone.backend.domain.emotion.dto.EmotionLogRequestDto;
import com.capstone.backend.domain.emotion.dto.EmotionLogResponseDto;
import com.capstone.backend.domain.emotion.service.EmotionLogService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/emotion-logs")
@RequiredArgsConstructor
public class EmotionLogController {

    private final EmotionLogService emotionLogService;

    @PostMapping
    public ResponseEntity<ApiResponse<EmotionLogResponseDto>> saveEmotionLog(
            @Valid @RequestBody EmotionLogRequestDto dto) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(ApiResponse.success(emotionLogService.saveEmotionLog(dto)));
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<ApiResponse<List<EmotionLogResponseDto>>> getEmotionLogs(
            @PathVariable Long userId) {
        return ResponseEntity.ok(ApiResponse.success(emotionLogService.getEmotionLogsByUserId(userId)));
    }

    @GetMapping("/user/{userId}/recent")
    public ResponseEntity<ApiResponse<List<EmotionLogResponseDto>>> getRecentEmotionLogs(
            @PathVariable Long userId,
            @RequestParam(defaultValue = "10") int count) {
        return ResponseEntity.ok(ApiResponse.success(emotionLogService.getRecentEmotionLogs(userId, count)));
    }
}
