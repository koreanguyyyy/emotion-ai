package com.capstone.backend.domain.intervention.dto;

import com.capstone.backend.domain.intervention.entity.EmotionAccuracy;
import com.capstone.backend.domain.intervention.entity.InterventionRating;
import lombok.Builder;

@Builder
public record FeedbackRequestDto(
        InterventionRating interventionRating,
        EmotionAccuracy emotionAccuracy,
        String feedbackText
) {}
