variable "image_name" {
  type    = string
  default = "cmapss-rul:latest"
}
variable "host_port" {
  type    = number
  default = 8000
}

resource "docker_container" "rul_api" {
  name  = "cmapss-rul-terraform"
  image = var.image_name
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
