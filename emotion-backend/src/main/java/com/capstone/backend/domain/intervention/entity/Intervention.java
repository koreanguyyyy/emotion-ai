package com.capstone.backend.domain.intervention.entity;

import com.capstone.backend.domain.emotion.entity.EmotionLog;
import com.capstone.backend.domain.user.entity.User;
import jakarta.persistence.*;
import lombok.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Entity
@Table(name = "interventions", indexes = {
        @Index(name = "idx_intervention_user_created", columnList = "user_id, created_at"),
        @Index(name = "idx_intervention_user_cooldown", columnList = "user_id, cooldown_expires_at"),
        @Index(name = "idx_intervention_rating", columnList = "intervention_rating")
})
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@EntityListeners(AuditingEntityListener.class)
public class Intervention {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "intervention_id")
    private Long interventionId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "emotion_log_id")
    private EmotionLog emotionLog;

    @Column(name = "intervention_type", length = 20)
    private String interventionType;

    @Column(name = "agent_message", columnDefinition = "TEXT")
    private String agentMessage;

    @Column(name = "tts_played")
    private Boolean ttsPlayed;

    @Column(name = "cooldown_expires_at")
    private LocalDateTime cooldownExpiresAt;

    @Enumerated(EnumType.STRING)
    @Column(name = "intervention_rating")
    private InterventionRating interventionRating;

    @Enumerated(EnumType.STRING)
    @Column(name = "emotion_accuracy")
    private EmotionAccuracy emotionAccuracy;

    @Column(name = "intervention_feedback_text", length = 500)
    private String interventionFeedbackText;

    @Column(name = "feedback_at")
    private LocalDateTime feedbackAt;

    @CreatedDate
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    public void submitFeedback(InterventionRating rating, EmotionAccuracy accuracy, String feedbackText) {
        this.interventionRating = rating;
        this.emotionAccuracy = accuracy;
        this.interventionFeedbackText = feedbackText;
        this.feedbackAt = LocalDateTime.now();
    }
}
