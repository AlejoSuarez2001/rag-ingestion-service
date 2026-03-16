# Si un usuario no esta en el grupo de docker debe usar `sudo docker`
docker := if `groups $USER` =~ "docker" { "docker" } else { "sudo docker" }
# Nombre del servicio principal
servicio := 'rag-ingestion'

# Al ejecutar `just` sin comandos muestra el listado de comandos disponibles
_default:
    @just --list --unsorted

#: Iniciar el servicio
deploy:
    {{docker}} compose up -d

#: Detener el servicio
down:
    {{docker}} compose down

#: Mostrar logs del servicio
logs:
    {{docker}} logs --follow {{servicio}}

#: Ejecuta bash dentro del container
bash:
    {{docker}} exec -it {{servicio}} sh

#: Construir la imagen
build:
    {{docker}} compose build

#: Reconstruir e iniciar el servicio
redeploy:
    {{docker}} compose up -d --build

#: Mostrar logs de postgres
logs-db:
    {{docker}} logs --follow postgres

#: Mostrar logs de qdrant
logs-qdrant:
    {{docker}} logs --follow qdrant
