package kr.clpeak

data class ResultEntry(
    val backend: String,
    val platform: String,
    val device: String,
    val driver: String,
    val test: String,
    val metric: String,
    val unit: String,
    val value: Float
)
