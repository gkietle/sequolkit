FROM maven:3.9.5 AS builder

ENV APPLICATION=slm-embed

COPY ./pom.xml ./

RUN --mount=type=cache,target=/.m2,id=${APPLICATION},uid=500,gid=500 \
    mvn -Dmaven.repo.local=/.m2/repository dependency:go-offline

COPY . .

RUN --mount=type=cache,target=/.m2,id=${APPLICATION},uid=500,gid=500 \
    mvn -Dmaven.repo.local=/.m2/repository -DskipTests package

FROM eclipse-temurin:21-jre AS final

COPY --from=builder /target/*.jar /app/application.jar

CMD ["java", "-jar", "/app/application.jar"]