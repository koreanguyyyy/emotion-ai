package com.capstone.backend.domain.recommendation.controller;

import com.capstone.backend.common.response.ApiResponse;
import com.capstone.backend.domain.recommendation.dto.RecommendationRequestDto;
import com.capstone.backend.domain.recommendation.dto.RecommendationResponseDto;
import com.capstone.backend.domain.recommendation.service.RecommendationService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/recommendations")
@RequiredArgsConstructor
public class RecommendationController {

    private final RecommendationService recommendationService;

    @PostMapping
    public ResponseEntity<ApiResponse<RecommendationResponseDto>> saveRecommendation(
            @Valid @RequestBody RecommendationRequestDto.Create dto) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(ApiResponse.success(recommendationService.saveRecommendation(dto)));
    }

    @GetMapping("/intervention/{interventionId}")
    public ResponseEntity<ApiResponse<List<RecommendationResponseDto>>> getRecommendations(
            @PathVariable Long interventionId) {
        return ResponseEntity.ok(ApiResponse.success(
                recommendationService.getRecommendationsByInterventionId(interventionId)));
    }

    @PostMapping("/{recommendationId}/feedback")
    public ResponseEntity<ApiResponse<RecommendationResponseDto>> submitFeedback(
            @PathVariable Long recommendationId,
            @RequestBody RecommendationRequestDto.Feedback dto) {
        return ResponseEntity.ok(ApiResponse.success(
                recommendationService.submitFeedback(recommendationId, dto)));
    }
}
