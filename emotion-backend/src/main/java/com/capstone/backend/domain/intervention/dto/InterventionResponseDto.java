package com.capstone.backend.domain.intervention.dto;

import com.capstone.backend.domain.intervention.entity.EmotionAccuracy;
import com.capstone.backend.domain.intervention.entity.Intervention;
import com.capstone.backend.domain.intervention.entity.InterventionRating;
import lombok.Builder;

import java.time.LocalDateTime;

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
        LocalDateTime createdAt
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
                .build();
    }
}
