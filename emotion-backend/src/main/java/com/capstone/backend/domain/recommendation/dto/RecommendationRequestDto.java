package com.capstone.backend.domain.recommendation.dto;

import com.capstone.backend.domain.recommendation.entity.RecommendationCategory;
import com.capstone.backend.domain.recommendation.entity.RelevanceRating;
import com.capstone.backend.domain.recommendation.entity.UserFeedback;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;

public class RecommendationRequestDto {

    @Builder
    public record Create(
            @NotNull Long interventionId,
            @NotNull RecommendationCategory category,
            String title,
            String artistOrAuthor,
            String externalId,
            String externalUrl,
            String thumbnailUrl,
            String reason
    ) {}

    @Builder
    public record Feedback(
            UserFeedback userFeedback,
            RelevanceRating relevanceRating,
            Boolean played,
            String feedbackText
    ) {}
}
