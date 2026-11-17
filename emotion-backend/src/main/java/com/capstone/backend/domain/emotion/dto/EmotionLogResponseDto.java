package com.capstone.backend.domain.emotion.dto;

import com.capstone.backend.domain.emotion.entity.EmotionLog;
import lombok.Builder;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Builder
public record EmotionLogResponseDto(
        Long emotionLogId,
        Long userId,
        String primaryEmotion,
        BigDecimal confidenceScore,
        String faceEmotion,
        BigDecimal faceConfidence,
        String speechText,
        String speechSentiment,
        BigDecimal vadStressScore,
        Integer finalStressLevel,
        Boolean sighDetected,
        String motionPattern,
        Boolean interventionTriggered,
        LocalDateTime detectedAt,
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
) {
    public static EmotionLogResponseDto from(EmotionLog log) {
        return EmotionLogResponseDto.builder()
                .emotionLogId(log.getEmotionLogId())
                .userId(log.getUser().getUserId())
                .primaryEmotion(log.getPrimaryEmotion())
                .confidenceScore(log.getConfidenceScore())
                .faceEmotion(log.getFaceEmotion())
                .faceConfidence(log.getFaceConfidence())
                .speechText(log.getSpeechText())
                .speechSentiment(log.getSpeechSentiment())
                .vadStressScore(log.getVadStressScore())
                .finalStressLevel(log.getFinalStressLevel())
                .sighDetected(log.getSighDetected())
                .motionPattern(log.getMotionPattern())
                .interventionTriggered(log.getInterventionTriggered())
                .detectedAt(log.getDetectedAt())
                .heartRate(log.getHeartRate())
                .restingHeartRate(log.getRestingHeartRate())
                .stressNeg(log.getStressNeg())
                .stressHr(log.getStressHr())
                .stressArousal(log.getStressArousal())
                .stressInstab(log.getStressInstab())
                .hrMode(log.getHrMode())
                .faceMethod(log.getFaceMethod())
                .modelEmotion(log.getModelEmotion())
                .logType(log.getLogType())
                .build();
    }
}
