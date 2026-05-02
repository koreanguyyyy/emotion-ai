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
}
