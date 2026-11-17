package com.capstone.backend.domain.intervention.dto;

import jakarta.validation.constraints.NotNull;
import lombok.Builder;

@Builder
public record InterventionRequestDto(
        @NotNull Long userId,
        @NotNull Long emotionLogId,
        String interventionType,
        String agentMessage,
        Boolean ttsPlayed,
        Integer cooldownMinutes
) {}
