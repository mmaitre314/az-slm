// Linux Spot VM driven through Run Command (no inbound access). Evicted VMs are deleted, along with
// their OS disk, NIC and public IP. Two layers of auto-shutdown:
//  1. An in-VM idle watchdog (installed by cloud-init) that deletes or deallocates the VM through
//     its system-assigned managed identity after a few hours without activity.
//  2. Azure's built-in daily auto-shutdown schedule (deallocate) as a fallback.

@description('VM name; also used to name its NIC, public IP, OS disk and auto-shutdown schedule.')
param name string

param location string

@description('VM size. Must be Spot-capable in the region and fit the regional Spot vCPU quota.')
param vmSize string

param subnetId string

param adminUsername string = 'azureuser'

@description('SSH public key. Required by Azure for Linux VMs, but never used: VMs are driven via Run Command.')
param sshPublicKey string

@description('cloud-init (#cloud-config) document, as plain text.')
param cloudInit string

param osDiskSizeGB int = 64

@description('Daily fallback shutdown time (HHmm, UTC). The schedule deallocates the VM.')
param dailyShutdownTime string = '0300'

param tags object = {}

var roles = {
  virtualMachineContributor: '9980e02c-c2be-4d73-94e8-173b1dc7cf3c'
  networkContributor: '4d97b98b-1d4f-4787-a291-c67834d212e7'
}

resource pip 'Microsoft.Network/publicIPAddresses@2024-05-01' = {
  name: 'pip-${name}'
  location: location
  tags: tags
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
    publicIPAddressVersion: 'IPv4'
  }
}

resource nic 'Microsoft.Network/networkInterfaces@2024-05-01' = {
  name: 'nic-${name}'
  location: location
  tags: tags
  properties: {
    enableAcceleratedNetworking: true
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: {
            id: subnetId
          }
          privateIPAllocationMethod: 'Dynamic'
          publicIPAddress: {
            id: pip.id
            properties: {
              deleteOption: 'Delete'
            }
          }
        }
      }
    ]
  }
}

resource vm 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: name
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    priority: 'Spot'
    evictionPolicy: 'Delete'
    billingProfile: {
      maxPrice: -1 // never evict on price; pay the current Spot price up to the pay-as-you-go price
    }
    hardwareProfile: {
      vmSize: vmSize
    }
    storageProfile: {
      diskControllerType: 'NVMe'
      imageReference: {
        publisher: 'Canonical'
        offer: 'ubuntu-24_04-lts'
        sku: 'server'
        version: 'latest'
      }
      osDisk: {
        name: 'osdisk-${name}'
        createOption: 'FromImage'
        caching: 'ReadWrite'
        diskSizeGB: osDiskSizeGB
        deleteOption: 'Delete'
        managedDisk: {
          storageAccountType: 'Premium_LRS'
        }
      }
    }
    osProfile: {
      computerName: name
      adminUsername: adminUsername
      customData: base64(cloudInit)
      linuxConfiguration: {
        disablePasswordAuthentication: true
        provisionVMAgent: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: nic.id
          properties: {
            primary: true
            deleteOption: 'Delete'
          }
        }
      ]
    }
    securityProfile: {
      securityType: 'TrustedLaunch'
      uefiSettings: {
        secureBootEnabled: true
        vTpmEnabled: true
      }
    }
  }
}

// The idle watchdog deletes/deallocates its own VM. Grant the VM identity just enough on the VM and
// the attached NIC and public IP, which are deleted with it (deleteOption: Delete).
resource vmRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vm.id, roles.virtualMachineContributor)
  scope: vm
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.virtualMachineContributor)
    principalId: vm.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource nicRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(nic.id, vm.id, roles.virtualMachineContributor)
  scope: nic
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.virtualMachineContributor)
    principalId: vm.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource pipRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(pip.id, vm.id, roles.networkContributor)
  scope: pip
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.networkContributor)
    principalId: vm.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Fallback: Azure's built-in daily auto-shutdown (deallocates the VM).
resource dailyShutdown 'Microsoft.DevTestLab/schedules@2018-09-15' = {
  name: 'shutdown-computevm-${name}'
  location: location
  tags: tags
  properties: {
    status: 'Enabled'
    taskType: 'ComputeVmShutdownTask'
    dailyRecurrence: {
      time: dailyShutdownTime
    }
    timeZoneId: 'UTC'
    targetResourceId: vm.id
    notificationSettings: {
      status: 'Disabled'
    }
  }
}

output vmName string = vm.name
