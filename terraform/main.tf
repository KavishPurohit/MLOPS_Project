variable "image_name" {
  type    = string
  default = "cmapss-rul-terraform:local"
}
variable "host_port" {
  type    = number
  default = 8000
}

resource "docker_image" "rul_api" {
  name = var.image_name
  build {
    context    = ".."
    dockerfile = "Dockerfile"
  }
}

resource "docker_container" "rul_api" {
  name  = "cmapss-rul-terraform"
  image = docker_image.rul_api.image_id
  ports {
    internal = 8000
    external = var.host_port
  }
  volumes {
    host_path      = abspath("${path.module}/../models")
    container_path = "/app/models"
    read_only      = true
  }
}

output "api_url" {
  value = "http://localhost:${var.host_port}"
}
