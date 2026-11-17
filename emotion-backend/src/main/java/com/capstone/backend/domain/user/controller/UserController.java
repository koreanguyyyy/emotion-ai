package com.capstone.backend.domain.user.controller;

import com.capstone.backend.common.response.ApiResponse;
import com.capstone.backend.domain.user.dto.UserRequestDto;
import com.capstone.backend.domain.user.dto.UserResponseDto;
import com.capstone.backend.domain.user.service.UserPreferenceService;
import com.capstone.backend.domain.user.service.UserService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;
    private final UserPreferenceService userPreferenceService;

    @PostMapping
    public ResponseEntity<ApiResponse<UserResponseDto.Info>> createUser(
            @Valid @RequestBody UserRequestDto.Create dto) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(ApiResponse.success(userService.createUser(dto)));
    }

    @GetMapping("/{userId}")
    public ResponseEntity<ApiResponse<UserResponseDto.Info>> getUser(@PathVariable Long userId) {
        return ResponseEntity.ok(ApiResponse.success(userService.getUserById(userId)));
    }

    @PutMapping("/{userId}/preferences")
    public ResponseEntity<ApiResponse<UserResponseDto.PreferenceInfo>> updatePreferences(
            @PathVariable Long userId,
            @RequestBody UserRequestDto.UpdatePreference dto) {
        return ResponseEntity.ok(ApiResponse.success(
                userPreferenceService.updatePreferences(userId, dto)));
    }
}
