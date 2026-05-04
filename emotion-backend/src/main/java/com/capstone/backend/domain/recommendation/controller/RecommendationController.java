package com.capstone.backend.domain.recommendation.controller;

import com.capstone.backend.common.response.ApiResponse;
import com.capstone.backend.domain.recommendation.dto.RecommendationRequestDto;
import com.capstone.backend.domain.recommendation.dto.RecommendationResponseDto;
import com.capstone.backend.domain.recommendation.service.RecommendationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "Recommendation", description = "추천 결과 관리 API")
@RestController
@RequestMapping("/api/recommendations")
@RequiredArgsConstructor
public class RecommendationController {

    private final RecommendationService recommendationService;

    @Operation(summary = "추천 결과 저장", description = "capstone-ai에서 호출. 개입에 연결된 추천 결과를 저장합니다.")
    @PostMapping
    public ResponseEntity<ApiResponse<RecommendationResponseDto>> saveRecommendation(
            @Valid @RequestBody RecommendationRequestDto.Create dto) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(ApiResponse.success(recommendationService.saveRecommendation(dto)));
    }

    @Operation(summary = "개입별 추천 목록 조회", description = "특정 개입(interventionId)에 연결된 추천 결과 목록을 반환합니다.")
    @GetMapping("/intervention/{interventionId}")
    public ResponseEntity<ApiResponse<List<RecommendationResponseDto>>> getRecommendations(
            @PathVariable Long interventionId) {
        return ResponseEntity.ok(ApiResponse.success(
                recommendationService.getRecommendationsByInterventionId(interventionId)));
    }

    @Operation(summary = "유저별 최근 추천 이력 조회", description = "capstone-ai에서 호출. 중복 추천 방지를 위해 userId 기준 최근 추천 이력을 최신순으로 반환합니다. size로 개수를 조절합니다 (기본 5).")
    @GetMapping("/user/{userId}/recent")
    public ResponseEntity<ApiResponse<List<RecommendationResponseDto>>> getRecentRecommendations(
            @PathVariable Long userId,
            @RequestParam(defaultValue = "5") int size) {
        return ResponseEntity.ok(ApiResponse.success(
                recommendationService.getRecentRecommendationsByUserId(userId, size)));
    }

    @Operation(summary = "추천 피드백 제출", description = "사용자가 추천 결과에 대한 평가(좋아요/싫어요/건너뜀, 관련도, 재생 여부)를 제출합니다.")
    @PostMapping("/{recommendationId}/feedback")
    public ResponseEntity<ApiResponse<RecommendationResponseDto>> submitFeedback(
            @PathVariable Long recommendationId,
            @RequestBody RecommendationRequestDto.Feedback dto) {
        return ResponseEntity.ok(ApiResponse.success(
                recommendationService.submitFeedback(recommendationId, dto)));
    }
}
