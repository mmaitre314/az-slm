// One benchmark VM: shared regional network + Spot VM with idle watchdog and daily auto-shutdown.
//   python3 scripts/deploy.py deploy infra/main.bicep --name <vmName> -p vmName=<vmName>

@description('VM name (also the value of the azslm-run tag used by `deploy.py teardown`).')
param vmName string

@description('Region for the network and VM. Defaults to eastus2: cheap Spot AMX capacity, both v6 and v7 Intel sizes.')
param location string = 'eastus2'

@description('Spot VM size. 16 vCPUs fits the default 20-vCPU regional Spot quota.')
param vmSize string = 'Standard_E16ds_v7'

@description('Spot (default) or Regular. Visual Studio (MSDN) subscriptions cannot deploy Spot VMs.')
@allowed([
  'Spot'
  'Regular'
])
param priority string = 'Spot'

@description('SSH public key (deploy.py generates a throwaway key when omitted).')
param sshPublicKey string

@description('Hours without activity before the in-VM watchdog acts.')
@minValue(1)
param idleHours int = 1

@description('What the idle watchdog does: delete frees the Spot quota; deallocate keeps the VM.')
@allowed([
  'delete'
  'deallocate'
])
param idleAction string = 'delete'

@description('Daily fallback shutdown time (HHmm, UTC). Empty: fallbackHours after the deployment hour, so the first fallback never lands minutes into a fresh benchmark run.')
param dailyShutdownTime string = ''

@description('With dailyShutdownTime empty: hours from the deployment hour to the daily fallback shutdown.')
@minValue(1)
@maxValue(23)
param fallbackHours int = 12

param deployHourUtc string = utcNow('HH')

var shutdownTime = empty(dailyShutdownTime)
  ? '${padLeft(string((int(deployHourUtc) + fallbackHours) % 24), 2, '0')}00'
  : dailyShutdownTime

@description('OS disk size. Models and results live on the local NVMe disks mounted at /mnt/data.')
param osDiskSizeGB int = 64

var cloudInit = replace(
  replace(loadTextContent('cloud-init.yaml'), '__IDLE_HOURS__', string(idleHours)),
  '__IDLE_ACTION__',
  idleAction
)

module network 'modules/network.bicep' = {
  name: 'network-${location}-${vmName}'
  params: {
    location: location
    tags: {
      'azslm-shared': 'network'
    }
  }
}

module vm 'modules/spot-vm.bicep' = {
  name: 'vm-${vmName}'
  params: {
    name: vmName
    location: location
    vmSize: vmSize
    priority: priority
    subnetId: network.outputs.subnetId
    sshPublicKey: sshPublicKey
    cloudInit: cloudInit
    osDiskSizeGB: osDiskSizeGB
    dailyShutdownTime: shutdownTime
    tags: {
      'azslm-run': vmName
    }
  }
}

output vmName string = vm.outputs.vmName
output dailyShutdownUtc string = shutdownTime
