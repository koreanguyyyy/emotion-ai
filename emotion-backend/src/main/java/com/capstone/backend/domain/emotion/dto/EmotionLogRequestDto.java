package com.capstone.backend.domain.emotion.dto;

import jakarta.validation.constraints.NotNull;
import lombok.Builder;

import java.math.BigDecimal;

@Builder
public record EmotionLogRequestDto(
        @NotNull Long userId,
        String faceEmotion,
        BigDecimal faceConfidence,
        String speechText,
        BigDecimal vadStressScore,
        Integer finalStressLevel,
        Boolean sighDetected,
        String motionPattern,
        Boolean interventionTriggered
) {}
