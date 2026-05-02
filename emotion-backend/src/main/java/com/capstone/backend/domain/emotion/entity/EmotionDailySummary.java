package com.capstone.backend.domain.emotion.entity;

import com.capstone.backend.common.converter.StringMapConverter;
import com.capstone.backend.domain.user.entity.User;
import jakarta.persistence.*;
import lombok.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Map;

@Entity
@Table(name = "emotion_daily_summary",
        uniqueConstraints = @UniqueConstraint(name = "uk_user_summary_date", columnNames = {"user_id", "summary_date"}))
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@EntityListeners(AuditingEntityListener.class)
public class EmotionDailySummary {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "summary_id")
    private Long summaryId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    @Column(name = "summary_date")
    private LocalDate summaryDate;

    @Column(name = "dominant_emotion", length = 30)
    private String dominantEmotion;

    @Convert(converter = StringMapConverter.class)
    @Column(name = "emotion_distribution", columnDefinition = "JSON")
    private Map<String, Object> emotionDistribution;

    @Column(name = "total_detections")
    private Integer totalDetections;

    @Column(name = "total_interventions")
    private Integer totalInterventions;

    @Column(name = "avg_confidence", precision = 5, scale = 4)
    private BigDecimal avgConfidence;

    @Column(name = "avg_stress_level", precision = 4, scale = 2)
    private BigDecimal avgStressLevel;

    @CreatedDate
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;
}
