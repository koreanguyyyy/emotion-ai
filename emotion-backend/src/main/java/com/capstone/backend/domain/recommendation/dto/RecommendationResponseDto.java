package com.capstone.backend.domain.recommendation.dto;

import com.capstone.backend.domain.recommendation.entity.Recommendation;
import com.capstone.backend.domain.recommendation.entity.RecommendationCategory;
import com.capstone.backend.domain.recommendation.entity.RelevanceRating;
import com.capstone.backend.domain.recommendation.entity.UserFeedback;
import lombok.Builder;

import java.time.LocalDateTime;

@Builder
public record RecommendationResponseDto(
        Long recommendationId,
        Long interventionId,
        RecommendationCategory category,
        String title,
        String artistOrAuthor,
        String externalId,
        String externalUrl,
        String thumbnailUrl,
        String reason,
        UserFeedback userFeedback,
        Boolean played,
        RelevanceRating relevanceRating,
        String feedbackText,
        LocalDateTime createdAt
) {
    public static RecommendationResponseDto from(Recommendation rec) {
        return RecommendationResponseDto.builder()
                .recommendationId(rec.getRecommendationId())
                .interventionId(rec.getIntervention().getInterventionId())
                .category(rec.getCategory())
                .title(rec.getTitle())
                .artistOrAuthor(rec.getArtistOrAuthor())
                .externalId(rec.getExternalId())
                .externalUrl(rec.getExternalUrl())
                .thumbnailUrl(rec.getThumbnailUrl())
                .reason(rec.getReason())
                .userFeedback(rec.getUserFeedback())
                .played(rec.getPlayed())
                .relevanceRating(rec.getRelevanceRating())
                .feedbackText(rec.getFeedbackText())
                .createdAt(rec.getCreatedAt())
                .build();
    }
}
