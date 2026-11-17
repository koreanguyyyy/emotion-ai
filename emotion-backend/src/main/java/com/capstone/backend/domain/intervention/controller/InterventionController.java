package com.capstone.backend.domain.intervention.controller;

import com.capstone.backend.common.response.ApiResponse;
import com.capstone.backend.domain.intervention.dto.FeedbackRequestDto;
import com.capstone.backend.domain.intervention.dto.InterventionRequestDto;
import com.capstone.backend.domain.intervention.dto.InterventionResponseDto;
import com.capstone.backend.domain.intervention.service.InterventionService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@Tag(name = "Intervention", description = "개입 이력 관리 API")
@RestController
@RequestMapping("/api/interventions")
@RequiredArgsConstructor
public class InterventionController {

    private final InterventionService interventionService;

    @Operation(summary = "개입 이력 저장", description = "capstone-ai에서 호출. 개입 이력을 저장하고 쿨다운을 설정합니다.")
    @PostMapping
    public ResponseEntity<ApiResponse<InterventionResponseDto>> saveIntervention(
            @Valid @RequestBody InterventionRequestDto dto) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(ApiResponse.success(interventionService.saveIntervention(dto)));
    }

    @Operation(summary = "최근 개입 이력 조회", description = "userId로 최신순 개입 이력을 조회합니다. size로 개수를 조절합니다 (기본 10). 각 개입에 연결된 추천 목록도 함께 반환합니다.")
    @GetMapping("/user/{userId}")
    public ResponseEntity<ApiResponse<List<InterventionResponseDto>>> getInterventions(
            @PathVariable Long userId,
            @RequestParam(defaultValue = "10") int size) {
        return ResponseEntity.ok(ApiResponse.success(
                interventionService.getRecentInterventionsByUserId(userId, size)));
    }

    @Operation(summary = "쿨다운 활성 여부 확인", description = "capstone-detection에서 호출. 해당 유저의 현재 쿨다운이 활성 상태인지 확인합니다.")
    @GetMapping("/user/{userId}/cooldown")
    public ResponseEntity<ApiResponse<Map<String, Boolean>>> checkCooldown(
            @PathVariable Long userId) {
        boolean active = interventionService.isCooldownActive(userId);
        return ResponseEntity.ok(ApiResponse.success(Map.of("cooldownActive", active)));
    }

    @Operation(summary = "개입 피드백 제출", description = "사용자가 개입에 대한 평가(유용도, 감정 정확도, 텍스트)를 제출합니다.")
    @PostMapping("/{interventionId}/feedback")
    public ResponseEntity<ApiResponse<InterventionResponseDto>> submitFeedback(
            @PathVariable Long interventionId,
            @RequestBody FeedbackRequestDto dto) {
        return ResponseEntity.ok(ApiResponse.success(
                interventionService.submitFeedback(interventionId, dto)));
    }
}
