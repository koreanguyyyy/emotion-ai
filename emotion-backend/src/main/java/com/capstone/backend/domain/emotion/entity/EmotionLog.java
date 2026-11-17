package com.capstone.backend.domain.emotion.entity;

import com.capstone.backend.domain.user.entity.User;
import jakarta.persistence.*;
import lombok.*;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "emotion_logs", indexes = {
        @Index(name = "idx_emotion_log_user_detected", columnList = "user_id, detected_at"),
        @Index(name = "idx_emotion_log_primary_emotion", columnList = "primary_emotion")
})
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
public class EmotionLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "emotion_log_id")
    private Long emotionLogId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    @Column(name = "primary_emotion", length = 30)
    private String primaryEmotion;

    @Column(name = "confidence_score", precision = 5, scale = 4)
    private BigDecimal confidenceScore;

    @Column(name = "face_emotion", length = 30)
    private String faceEmotion;

    @Column(name = "face_confidence", precision = 5, scale = 4)
    private BigDecimal faceConfidence;

    @Column(name = "speech_text", columnDefinition = "TEXT")
    private String speechText;

    @Column(name = "speech_sentiment", length = 30)
    private String speechSentiment;

    @Column(name = "vad_stress_score", precision = 4, scale = 2)
    private BigDecimal vadStressScore;

    @Column(name = "final_stress_level")
    private Integer finalStressLevel;

    @Builder.Default
    @Column(name = "sigh_detected")
    private Boolean sighDetected = false;

    @Column(name = "motion_pattern", length = 30)
    private String motionPattern;

    @Column(name = "intervention_triggered")
    private Boolean interventionTriggered;

    @Column(name = "detected_at")
    private LocalDateTime detectedAt;

    // ── 2026-11-03 추가: fusion 버전(표정 + 웹캠 심박) 정보. 모두 비어 있어도 됨(1학기 감지 프로그램 호환) ──

    /** 웹캠 심박 (bpm, 품질 기준 통과 값만) */
    @Column(name = "heart_rate", precision = 5, scale = 1)
    private BigDecimal heartRate;

    /** 개인 평상시 심박 (bpm) */
    @Column(name = "resting_heart_rate", precision = 5, scale = 1)
    private BigDecimal restingHeartRate;

    /** 스트레스 지수 구성: 부정 표정 정도 0~1 */
    @Column(name = "stress_neg", precision = 4, scale = 3)
    private BigDecimal stressNeg;

    /** 스트레스 지수 구성: 심박 상승 정도 0~1 (심박 모르면 null) */
    @Column(name = "stress_hr", precision = 4, scale = 3)
    private BigDecimal stressHr;

    /** 스트레스 지수 구성: 표정 각성 정도 0~1 */
    @Column(name = "stress_arousal", precision = 4, scale = 3)
    private BigDecimal stressArousal;

    /** 스트레스 지수 구성: 감정 기복 정도 0~1 */
    @Column(name = "stress_instab", precision = 4, scale = 3)
    private BigDecimal stressInstab;

    /** fusion / face_only / face_only_gap / calibrating */
    @Column(name = "hr_mode", length = 20)
    private String hrMode;

    /** 표정 계산 방식: rule / model */
    @Column(name = "face_method", length = 10)
    private String faceMethod;

    /** 학습된 표정 모델의 감정 라벨 (비교용) */
    @Column(name = "model_emotion", length = 30)
    private String modelEmotion;

    /** periodic(1분마다) / threshold(개입 기준 초과) */
    @Column(name = "log_type", length = 20)
    private String logType;
}
