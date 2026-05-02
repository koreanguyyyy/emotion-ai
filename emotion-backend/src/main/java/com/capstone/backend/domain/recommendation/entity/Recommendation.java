package com.capstone.backend.domain.recommendation.entity;

import com.capstone.backend.domain.intervention.entity.Intervention;
import jakarta.persistence.*;
import lombok.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;

@Entity
@Table(name = "recommendations", indexes = {
        @Index(name = "idx_recommendation_intervention", columnList = "intervention_id"),
        @Index(name = "idx_recommendation_category", columnList = "category"),
        @Index(name = "idx_recommendation_user_feedback", columnList = "user_feedback"),
        @Index(name = "idx_recommendation_relevance", columnList = "relevance_rating")
})
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@EntityListeners(AuditingEntityListener.class)
public class Recommendation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "recommendation_id")
    private Long recommendationId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "intervention_id")
    private Intervention intervention;

    @Enumerated(EnumType.STRING)
    @Column(name = "category")
    private RecommendationCategory category;

    @Column(length = 200)
    private String title;

    @Column(name = "artist_or_author", length = 200)
    private String artistOrAuthor;

    @Column(name = "external_id")
    private String externalId;

    @Column(name = "external_url")
    private String externalUrl;

    @Column(name = "thumbnail_url")
    private String thumbnailUrl;

    @Column(columnDefinition = "TEXT")
    private String reason;

    @Enumerated(EnumType.STRING)
    @Column(name = "user_feedback")
    private UserFeedback userFeedback;

    private Boolean played;

    @Enumerated(EnumType.STRING)
    @Column(name = "relevance_rating")
    private RelevanceRating relevanceRating;

    @Column(name = "feedback_text", length = 500)
    private String feedbackText;

    @CreatedDate
    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    public void submitFeedback(UserFeedback feedback, RelevanceRating relevance, Boolean played, String feedbackText) {
        this.userFeedback = feedback;
        this.relevanceRating = relevance;
        if (played != null) this.played = played;
        this.feedbackText = feedbackText;
    }
}
