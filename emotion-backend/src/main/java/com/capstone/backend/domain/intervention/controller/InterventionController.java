package com.capstone.backend.domain.intervention.controller;

import com.capstone.backend.common.response.ApiResponse;
import com.capstone.backend.domain.intervention.dto.FeedbackRequestDto;
import com.capstone.backend.domain.intervention.dto.InterventionRequestDto;
import com.capstone.backend.domain.intervention.dto.InterventionResponseDto;
import com.capstone.backend.domain.intervention.service.InterventionService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/interventions")
@RequiredArgsConstructor
public class InterventionController {

    private final InterventionService interventionService;

    @PostMapping
    public ResponseEntity<ApiResponse<InterventionResponseDto>> saveIntervention(
            @Valid @RequestBody InterventionRequestDto dto) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(ApiResponse.success(interventionService.saveIntervention(dto)));
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<ApiResponse<List<InterventionResponseDto>>> getInterventions(
            @PathVariable Long userId) {
        return ResponseEntity.ok(ApiResponse.success(interventionService.getInterventionsByUserId(userId)));
    }

    @GetMapping("/user/{userId}/cooldown")
    public ResponseEntity<ApiResponse<Map<String, Boolean>>> checkCooldown(
            @PathVariable Long userId) {
        boolean active = interventionService.isCooldownActive(userId);
        return ResponseEntity.ok(ApiResponse.success(Map.of("cooldownActive", active)));
    }

    @PostMapping("/{interventionId}/feedback")
    public ResponseEntity<ApiResponse<InterventionResponseDto>> submitFeedback(
            @PathVariable Long interventionId,
            @RequestBody FeedbackRequestDto dto) {
        return ResponseEntity.ok(ApiResponse.success(
                interventionService.submitFeedback(interventionId, dto)));
    }
}
