param(
    [Parameter(Mandatory = $true)]
    [string]$Repository,
    [Parameter(Mandatory = $true)]
    [string]$AllowedRoot,
    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,
    [Parameter(Mandatory = $true)]
    [string]$SourceHost,
    [Parameter(Mandatory = $true)]
    [string]$SourcePath,
    [int]$ExpectedMinimumFiles = 100000
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Write-AtomicJson {
    param([string]$Path, [hashtable]$Value)
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Assert-WithinRoot {
    param([string]$Candidate, [string]$Root)
    $candidateFull = [IO.Path]::GetFullPath($Candidate)
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $prefix = $rootFull + [IO.Path]::DirectorySeparatorChar
    if (-not $candidateFull.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "destination must be a strict child of the allowed root"
    }
    return $candidateFull
}

$repositoryFull = [IO.Path]::GetFullPath($Repository)
$allowedRootFull = [IO.Path]::GetFullPath($AllowedRoot)
$destinationFull = Assert-WithinRoot -Candidate $DestinationRoot -Root $allowedRootFull
if ($SourceHost -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$') {
    throw "unsafe SSH host alias"
}
if ($SourcePath -notmatch '^/[A-Za-z0-9._/-]+$') {
    throw "unsafe remote source path"
}
if (-not (Test-Path -LiteralPath (Join-Path $repositoryFull ".git"))) {
    throw "repository is not a Git checkout"
}

$gitCommit = (& git -C $repositoryFull rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "cannot resolve repository commit" }
$dirty = (& git -C $repositoryFull status --porcelain) -join "`n"
if ($LASTEXITCODE -ne 0 -or $dirty) { throw "repository must be clean" }

New-Item -ItemType Directory -Path $destinationFull -Force | Out-Null
$statePath = Join-Path $destinationFull "MIRROR_STATE.json"
$archivePath = Join-Path $destinationFull "dataset.tar.partial"
$stderrPath = Join-Path $destinationFull "TRANSFER_STDERR.log"
$datasetPath = Join-Path $destinationFull "dataset"
if (Test-Path -LiteralPath $datasetPath) {
    throw "destination dataset already exists; refusing to overwrite it"
}
if (Test-Path -LiteralPath $archivePath) {
    throw "partial archive already exists; inspect it before retrying"
}

$started = [DateTimeOffset]::Now
$sourceParent = $SourcePath.Substring(0, $SourcePath.LastIndexOf('/'))
if (-not $sourceParent) { $sourceParent = "/" }
$sourceName = $SourcePath.Substring($SourcePath.LastIndexOf('/') + 1)
$remoteCommand = "ionice -c3 nice -n 19 tar -C '$sourceParent' -cf - '$sourceName'"

Write-AtomicJson -Path $statePath -Value @{
    schema = "final-unsb-paper-local-data-mirror-state-v1"
    status = "TRANSFERRING_IDLE_IO_PRIORITY"
    pid = $PID
    repository_commit = $gitCommit
    source_host = $SourceHost
    source_path = $SourcePath
    destination_root = $destinationFull
    archive_path = $archivePath
    started_at = $started.ToString("o")
    source_mutated = $false
    confirmation20_evaluated = $false
}

try {
    $ssh = (Get-Command ssh).Source
    & $ssh -o BatchMode=yes -o ConnectTimeout=30 $SourceHost $remoteCommand `
        > $archivePath 2> $stderrPath
    if ($LASTEXITCODE -ne 0) {
        throw "SSH/tar transfer exited with code $LASTEXITCODE"
    }
    if ((Get-Item -LiteralPath $archivePath).Length -lt 1GB) {
        throw "received archive is unexpectedly small"
    }

    $tar = (Get-Command tar).Source
    & $tar -tf $archivePath *> $null
    if ($LASTEXITCODE -ne 0) { throw "received archive failed tar integrity check" }
    & $tar -xf $archivePath -C $destinationFull
    if ($LASTEXITCODE -ne 0) { throw "received archive failed extraction" }
    if (-not (Test-Path -LiteralPath $datasetPath -PathType Container)) {
        throw "archive did not create the expected dataset directory"
    }
    $files = (
        Get-ChildItem -LiteralPath $datasetPath -Recurse -File | Measure-Object
    ).Count
    if ($files -lt $ExpectedMinimumFiles) {
        throw "extracted dataset contains only $files files"
    }
    Write-AtomicJson -Path $statePath -Value @{
        schema = "final-unsb-paper-local-data-mirror-state-v1"
        status = "MIRROR_COMPLETE_AWAITING_MANIFEST_HASH_GATE"
        pid = $PID
        repository_commit = $gitCommit
        source_host = $SourceHost
        source_path = $SourcePath
        destination_root = $destinationFull
        archive_path = $archivePath
        archive_bytes = (Get-Item -LiteralPath $archivePath).Length
        extracted_file_count = $files
        started_at = $started.ToString("o")
        completed_at = [DateTimeOffset]::Now.ToString("o")
        source_mutated = $false
        confirmation20_evaluated = $false
    }
}
catch {
    Write-AtomicJson -Path $statePath -Value @{
        schema = "final-unsb-paper-local-data-mirror-state-v1"
        status = "MIRROR_FAILED_REVIEW_REQUIRED"
        pid = $PID
        repository_commit = $gitCommit
        source_host = $SourceHost
        source_path = $SourcePath
        destination_root = $destinationFull
        archive_path = $archivePath
        started_at = $started.ToString("o")
        failed_at = [DateTimeOffset]::Now.ToString("o")
        error = $_.Exception.Message
        source_mutated = $false
        confirmation20_evaluated = $false
    }
    throw
}
