package com.capstone.backend.domain.user.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Builder;

import java.util.List;

public class UserRequestDto {

    @Builder
    public record Create(
            @NotBlank @Size(max = 50) String loginId,
            @NotBlank String password,
            @Email @NotBlank @Size(max = 100) String email,
            @NotBlank @Size(max = 50) String nickname,
            String address,
            String addressDetail,
            String zipcode
    ) {}

    @Builder
    public record UpdatePreference(
            List<String> preferredMusicGenres,
            List<String> preferredMovieGenres,
            List<String> preferredBookGenres,
            List<String> preferredPlaceTypes,
            Boolean ttsEnabled,
            Integer cooldownMinutes
    ) {}
}
