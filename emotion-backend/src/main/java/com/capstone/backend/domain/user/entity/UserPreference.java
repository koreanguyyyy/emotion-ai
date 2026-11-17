package com.capstone.backend.domain.user.entity;

import com.capstone.backend.common.converter.StringListConverter;
import jakarta.persistence.*;
import lombok.*;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "user_preferences")
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@EntityListeners(AuditingEntityListener.class)
public class UserPreference {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "preference_id")
    private Long preferenceId;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", unique = true)
    private User user;

    @Convert(converter = StringListConverter.class)
    @Column(name = "preferred_music_genres", columnDefinition = "JSON")
    private List<String> preferredMusicGenres;

    @Convert(converter = StringListConverter.class)
    @Column(name = "preferred_movie_genres", columnDefinition = "JSON")
    private List<String> preferredMovieGenres;

    @Convert(converter = StringListConverter.class)
    @Column(name = "preferred_book_genres", columnDefinition = "JSON")
    private List<String> preferredBookGenres;

    @Convert(converter = StringListConverter.class)
    @Column(name = "preferred_place_types", columnDefinition = "JSON")
    private List<String> preferredPlaceTypes;

    @Builder.Default
    @Column(name = "tts_enabled")
    private Boolean ttsEnabled = true;

    @Builder.Default
    @Column(name = "cooldown_minutes")
    private Integer cooldownMinutes = 30;

    @LastModifiedDate
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    public void update(List<String> musicGenres, List<String> movieGenres,
                       List<String> bookGenres, List<String> placeTypes,
                       Boolean ttsEnabled, Integer cooldownMinutes) {
        this.preferredMusicGenres = musicGenres;
        this.preferredMovieGenres = movieGenres;
        this.preferredBookGenres = bookGenres;
        this.preferredPlaceTypes = placeTypes;
        if (ttsEnabled != null) this.ttsEnabled = ttsEnabled;
        if (cooldownMinutes != null) this.cooldownMinutes = cooldownMinutes;
    }
}
