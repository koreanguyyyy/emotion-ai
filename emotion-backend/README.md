# emotion-backend

Spring Boot 기반 백엔드 서버

## 모듈 구성
| 패키지 | 설명 |
|--------|------|
| `user/` | 회원/인증 (JWT) |
| `history/` | 추천 이력 저장 |
| `config/` | Redis, Security 설정 |
| `common/` | 공통 응답/예외 처리 |

## 실행 방법
```bash
./gradlew bootRun
```

## 기술 스택
- Java 17 + Spring Boot 3.x
- MySQL + Redis
- JWT 인증
