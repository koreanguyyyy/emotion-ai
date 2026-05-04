package com.capstone.backend.domain.intervention.dto;

import com.capstone.backend.domain.intervention.entity.EmotionAccuracy;
import com.capstone.backend.domain.intervention.entity.Intervention;
import com.capstone.backend.domain.intervention.entity.InterventionRating;
import com.capstone.backend.domain.recommendation.dto.RecommendationResponseDto;
import lombok.Builder;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Builder
public record InterventionResponseDto(
        Long interventionId,
        Long userId,
        Long emotionLogId,
        String interventionType,
        String agentMessage,
        Boolean ttsPlayed,
        LocalDateTime cooldownExpiresAt,
        InterventionRating interventionRating,
        EmotionAccuracy emotionAccuracy,
        String interventionFeedbackText,
        LocalDateTime feedbackAt,
        LocalDateTime createdAt,
        List<RecommendationResponseDto> recommendations
) {
    public static InterventionResponseDto from(Intervention intervention) {
        return InterventionResponseDto.builder()
                .interventionId(intervention.getInterventionId())
                .userId(intervention.getUser().getUserId())
                .emotionLogId(intervention.getEmotionLog().getEmotionLogId())
                .interventionType(intervention.getInterventionType())
                .agentMessage(intervention.getAgentMessage())
                .ttsPlayed(intervention.getTtsPlayed())
                .cooldownExpiresAt(intervention.getCooldownExpiresAt())
                .interventionRating(intervention.getInterventionRating())
                .emotionAccuracy(intervention.getEmotionAccuracy())
                .interventionFeedbackText(intervention.getInterventionFeedbackText())
                .feedbackAt(intervention.getFeedbackAt())
                .createdAt(intervention.getCreatedAt())
                .recommendations(intervention.getRecommendations().stream()
                        .map(RecommendationResponseDto::from)
                        .collect(Collectors.toList()))
                .build();
    }
}
