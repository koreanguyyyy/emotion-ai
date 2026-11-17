package com.capstone.backend.domain.user.dto;

import com.capstone.backend.domain.user.entity.User;
import com.capstone.backend.domain.user.entity.UserPreference;
import lombok.Builder;

import java.time.LocalDateTime;
import java.util.List;

public class UserResponseDto {

    @Builder
    public record Info(
            Long userId,
            String loginId,
            String email,
            String nickname,
            String address,
            String addressDetail,
            String zipcode,
            LocalDateTime createdAt
    ) {
        public static Info from(User user) {
            return Info.builder()
                    .userId(user.getUserId())
                    .loginId(user.getLoginId())
                    .email(user.getEmail())
                    .nickname(user.getNickname())
                    .address(user.getAddress())
                    .addressDetail(user.getAddressDetail())
                    .zipcode(user.getZipcode())
                    .createdAt(user.getCreatedAt())
                    .build();
        }
    }

    @Builder
    public record PreferenceInfo(
            Long preferenceId,
            Long userId,
            List<String> preferredMusicGenres,
            List<String> preferredMovieGenres,
            List<String> preferredBookGenres,
            List<String> preferredPlaceTypes,
            Boolean ttsEnabled,
            Integer cooldownMinutes
    ) {
        public static PreferenceInfo from(UserPreference pref) {
            return PreferenceInfo.builder()
                    .preferenceId(pref.getPreferenceId())
                    .userId(pref.getUser().getUserId())
                    .preferredMusicGenres(pref.getPreferredMusicGenres())
                    .preferredMovieGenres(pref.getPreferredMovieGenres())
                    .preferredBookGenres(pref.getPreferredBookGenres())
                    .preferredPlaceTypes(pref.getPreferredPlaceTypes())
                    .ttsEnabled(pref.getTtsEnabled())
                    .cooldownMinutes(pref.getCooldownMinutes())
                    .build();
        }
    }
}
