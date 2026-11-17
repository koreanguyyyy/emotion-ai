package com.capstone.backend.domain.emotion.dto;

import jakarta.validation.constraints.NotNull;
import lombok.Builder;

import java.math.BigDecimal;

@Builder
public record EmotionLogRequestDto(
        @NotNull Long userId,
        String primaryEmotion,
        BigDecimal confidenceScore,
        String faceEmotion,
        BigDecimal faceConfidence,
        String speechText,
        BigDecimal vadStressScore,
        Integer finalStressLevel,
        Boolean sighDetected,
        String motionPattern,
        Boolean interventionTriggered,
        BigDecimal heartRate,
        BigDecimal restingHeartRate,
        BigDecimal stressNeg,
        BigDecimal stressHr,
        BigDecimal stressArousal,
        BigDecimal stressInstab,
        String hrMode,
        String faceMethod,
        String modelEmotion,
        String logType
) {}
