# Installs the latest Antares release on Windows.
#
#   irm https://raw.githubusercontent.com/Madushan996/Antares-Programming-Language/main/install.ps1 | iex
#
# Antares goes into %LOCALAPPDATA%\Programs\Antares, and that folder is added
# to your user PATH.
$ErrorActionPreference = 'Stop'

$repo = 'Madushan996/Antares-Programming-Language'
$name = 'antares-windows-x86_64'
$dir = Join-Path $env:LOCALAPPDATA 'Programs\Antares'
$url = "https://github.com/$repo/releases/latest/download/$name.zip"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ('antares-' + [guid]::NewGuid())

New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    Write-Host "Downloading $url"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile (Join-Path $tmp 'antares.zip') -UseBasicParsing
    Expand-Archive -Path (Join-Path $tmp 'antares.zip') -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Copy-Item -Path (Join-Path $tmp "$name\antares.exe") -Destination (Join-Path $dir 'antares.exe') -Force
}
finally {
    Remove-Item -Recurse -Force -Path $tmp -ErrorAction SilentlyContinue
}

$exe = Join-Path $dir 'antares.exe'
Write-Host ("Installed " + (& $exe version) + " to $exe")

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($null -eq $userPath) { $userPath = '' }
if (-not (($userPath -split ';') -contains $dir)) {
    $newPath = (($userPath.TrimEnd(';')) + ';' + $dir).TrimStart(';')
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    $env:Path = "$env:Path;$dir"
    Write-Host "Added $dir to your PATH. Open a new terminal, then try: antares version"
}
